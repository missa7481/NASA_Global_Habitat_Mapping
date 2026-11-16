#!/usr/bin/env python3
"""Shared data, holdout designs, models and metrics for the RSI research loop.

Task (unchanged from HiCo-Fuse): per 0.05-degree report cell, the multinomial composition of
Culicidae reports over {Ae. albopictus, Ae. aegypti, Culex, Anopheles, other}. Conditioning on
'a mosquito was reported here' cancels species-agnostic observer effort (target-group logic).

Holdout designs (each = list of splits; a split = train cells + train counts, test cells + test counts):
  random5        random cell 5-fold                     (interpolation reference, optimistic)
  block5/10/20   5/10/20-degree spatial blocks, 5 folds (scale curve)
  loro           leave-one-region-out, 6 regions
  env            leave-one-environmental-cluster-out (k-means, k=5, on climate PCs)
  zone           leave-one-climate-zone-out (Koppen-like: tropical / arid / temperate / cold)
  temporal       train 2022-2024 reports, test 2025 reports (all cells)
  spatiotemp     2025 reports in held-out 10-degree folds, trained on 2022-2024 elsewhere
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hicofuse.dataset import CLASSES as CLS5, cell_of, classify  # noqa: E402
from hicofuse.models import Net, multinomial_nll  # noqa: E402
from hicofuse.train import Problem, Store, fit as fit_hico  # noqa: E402

ART = ROOT / "artifacts/rsi"
ART.mkdir(parents=True, exist_ok=True)
DEV = "cuda"
CL4 = ["Ae_albopictus", "Ae_aegypti", "Culex", "Anopheles"]
REGIONS = ["N America", "Latin America", "Europe", "Africa", "Asia", "Oceania"]
ZONES = ["Tropical", "Arid", "Temperate", "Cold"]
MODS = ["Climate", "Landscape", "Human", "AlphaEarth"]


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, store: bool = True):
        self.st = Store() if store else None
        d = np.load(ROOT / "artifacts/hicofuse/dataset.npz")
        info = json.loads((ROOT / "artifacts/hicofuse/dataset_meta.json").read_text())
        self.feats, self.cols = info["features"], info["modality_cols"]
        self.obs_idx, self.Y = d["obs_idx"], d["Y"].astype(np.float32)
        self.fold10 = d["fold"]
        rc = d["rc"][self.obs_idx]
        self.rc = rc
        self.lat = 85 - (rc[:, 0] + 0.5) * 0.05
        self.lon = -180 + (rc[:, 1] + 0.5) * 0.05
        if store:
            self.pos = self.st.pos[self.obs_idx]
            self.X = self.st.X[self.pos]
        self.raw = self._raw(d)
        self.region = self._regions()
        self.zone = self._zones()
        self.Yearly, self.Ylate = self._periods(d)

    def _raw(self, d):
        X = np.load(ROOT / "artifacts/hicofuse/store/X.npy", mmap_mode="r")
        x = np.asarray(X[self.obs_idx], dtype=np.float64)
        return pd.DataFrame(x, columns=self.feats)

    def _regions(self):
        lat, lon = self.lat, self.lon
        r = np.full(len(lat), "Asia", dtype=object)
        am = lon < -30
        r[am & (lat >= 23.5)] = "N America"
        r[am & (lat < 23.5)] = "Latin America"
        eu = (~am) & (lon < 45) & (lat >= 35)
        r[eu] = "Europe"
        af = (~am) & (lon >= -20) & (lon <= 52) & (lat < 35) & ~((lon >= 35) & (lat >= 12))
        r[af] = "Africa"
        oc = ((lon >= 110) & (lat < -10)) | (lon >= 160) | (lon < -150)
        r[oc] = "Oceania"
        return r

    def _zones(self):
        t, p, tcold = self.raw["bio19_1"], self.raw["bio19_12"], self.raw["bio19_11"]
        tmin = self.raw["bio19_6"]
        z = np.full(len(t), "Temperate", dtype=object)
        z[(tcold < -3).to_numpy()] = "Cold"
        z[(tmin >= 18).to_numpy() | (tcold >= 18).to_numpy()] = "Tropical"
        z[(p < 20 * (t + 7)).to_numpy()] = "Arid"                  # Koppen B criterion (approx.)
        return z

    def _periods(self, d):
        occ = pd.read_csv(ROOT.parent / "data/raw/inat_culicidae_2022_2025/occurrences.csv")
        occ = occ[occ["coordinateUncertaintyInMeters"].le(2000)].copy()
        occ["cls"] = classify(occ)
        r, c = cell_of(occ["decimalLatitude"], occ["decimalLongitude"])
        key = r * 7200 + c
        okey = self.rc[:, 0].astype(np.int64) * 7200 + self.rc[:, 1]
        o = np.argsort(okey)
        p = np.clip(np.searchsorted(okey[o], key), 0, len(okey) - 1)
        hit = okey[o][p] == key
        occ = occ[hit].assign(cell=o[p][hit])
        out = []
        for late in (False, True):
            sub = occ[(occ["year"] == 2025) == late]
            Yp = np.zeros_like(self.Y)
            np.add.at(Yp, (sub["cell"].to_numpy(), sub["cls"].to_numpy()), 1)
            out.append(Yp)
        return out

    def env_clusters(self, k=5, seed=0):
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        xc = self.X[:, self.cols["Climate"]]
        z = PCA(6, random_state=seed).fit_transform(xc)
        return KMeans(k, n_init=10, random_state=seed).fit_predict(z)

    def coords(self):
        """Sphere-aware coordinate encoding (3-D unit vector + multi-frequency sinusoids)."""
        la, lo = np.radians(self.lat), np.radians(self.lon)
        f = [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
        for w in (1, 2, 4, 8, 16):
            f += [np.sin(w * la), np.cos(w * la), np.sin(w * lo), np.cos(w * lo)]
        return np.stack(f, 1).astype(np.float32)


def designs(D: Data) -> dict:
    n = len(D.Y)
    allc = np.arange(n)
    out = {}
    rng = np.random.default_rng(11)
    rf = rng.permutation(n) % 5
    out["random5"] = [(allc[rf != k], D.Y[rf != k], allc[rf == k], D.Y[rf == k], f"fold{k}") for k in range(5)]
    for deg in (5, 10, 20):
        if deg == 10:
            f = D.fold10
        else:
            b = (D.rc[:, 0] // (deg * 20)).astype(np.int64) * 1000 + D.rc[:, 1] // (deg * 20)
            ub = np.unique(b)
            fb = dict(zip(ub, np.random.default_rng(20261114 + deg).permutation(len(ub)) % 5))
            f = np.array([fb[x] for x in b])
        out[f"block{deg}"] = [(allc[f != k], D.Y[f != k], allc[f == k], D.Y[f == k], f"fold{k}") for k in range(5)]
    out["loro"] = [(allc[D.region != r], D.Y[D.region != r], allc[D.region == r], D.Y[D.region == r], r) for r in REGIONS]
    ec = D.env_clusters()
    out["env"] = [(allc[ec != k], D.Y[ec != k], allc[ec == k], D.Y[ec == k], f"envC{k}") for k in range(5)]
    out["zone"] = [(allc[D.zone != z], D.Y[D.zone != z], allc[D.zone == z], D.Y[D.zone == z], z) for z in ZONES]
    e, l_ = D.Yearly.sum(1) > 0, D.Ylate.sum(1) > 0
    out["temporal"] = [(allc[e], D.Yearly[e], allc[l_], D.Ylate[l_], "2025")]
    out["spatiotemp"] = [(allc[e & (D.fold10 != k)], D.Yearly[e & (D.fold10 != k)],
                          allc[l_ & (D.fold10 == k)], D.Ylate[l_ & (D.fold10 == k)], f"fold{k}") for k in range(5)]
    return out


def inner_split(D: Data, tr: np.ndarray, seed: int, frac=0.15):
    """Early-stopping split of a training set by 10-degree blocks."""
    b = (D.rc[tr, 0] // 200).astype(np.int64) * 100 + D.rc[tr, 1] // 200
    ub = np.unique(b)
    vb = np.random.default_rng(seed).choice(ub, max(1, int(round(frac * len(ub)))), replace=False)
    v = np.isin(b, vb)
    return ~v, v


# ----------------------------------------------------------------------------- metrics
def probs_from_eta(eta: np.ndarray) -> np.ndarray:
    z = np.c_[eta, np.zeros(len(eta))].astype(np.float64)
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def cell_nll(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return -(Y * np.log(np.clip(P, 1e-9, 1))).sum(1) / Y.sum(1)


def metrics(P: np.ndarray, Y: np.ndarray) -> dict:
    out = {"nll": float(-(Y * np.log(np.clip(P, 1e-9, 1))).sum() / Y.sum()), "n_cells": int(len(Y))}
    aucs = []
    for s, nm in enumerate(CL4):
        y = Y[:, s] > 0
        a = roc_auc_score(y, P[:, s]) if 5 <= y.sum() <= len(y) - 5 else np.nan
        out[f"AUC_{nm}"] = a
        aucs.append(a)
    out["AUC_mean"] = float(np.nanmean(aucs)) if np.isfinite(aucs).any() else np.nan
    # calibration of class share (record-weighted, 10 bins), averaged over the 4 target classes
    eces = []
    for s in range(4):
        w = Y.sum(1)
        obs = Y[:, s] / w
        b = np.minimum((P[:, s] * 10).astype(int), 9)
        e = sum(abs(np.average(P[b == k, s], weights=w[b == k]) - np.average(obs[b == k], weights=w[b == k])) * w[b == k].sum()
                for k in range(10) if (b == k).any()) / w.sum()
        eces.append(e)
    out["ECE"] = float(np.mean(eces))
    return out


# ----------------------------------------------------------------------------- models
class FlatMLP(nn.Module):
    def __init__(self, d_in, hidden=64, depth=2, p=0.1, linear=False):
        super().__init__()
        if linear:
            self.f = nn.Linear(d_in, 4)
        else:
            layers, d = [], d_in
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(p)]
                d = hidden
            self.f = nn.Sequential(*layers, nn.Linear(d, 4))

    def forward(self, x):
        return self.f(x)


def group_risks(eta, Y, g, n_g):
    logits = torch.cat([eta, torch.zeros(len(eta), 1, device=eta.device)], 1)
    ll = -(Y * F.log_softmax(logits, 1)).sum(1)
    num = torch.zeros(n_g, device=eta.device).index_add(0, g, ll)
    den = torch.zeros(n_g, device=eta.device).index_add(0, g, Y.sum(1))
    return num / den.clamp_min(1)


def fit_flat(Xtr, Ytr, Xva, Yva, seed=0, linear=False, groups=None, penalty=None, lam=0.0, wd=1e-4,
             max_epochs=1000, patience=60, hidden=64):
    """Full-batch AdamW with early stopping on block-held-out validation NLL.
    penalty: None | 'vrex' (variance of group risks) | 'dro' (worst-group risk, soft via logsumexp)."""
    torch.manual_seed(seed)
    xt, yt = torch.as_tensor(Xtr, device=DEV), torch.as_tensor(Ytr, device=DEV)
    xv, yv = torch.as_tensor(Xva, device=DEV), torch.as_tensor(Yva, device=DEV)
    net = FlatMLP(Xtr.shape[1], linear=linear, hidden=hidden).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=wd)
    if groups is not None:
        ug, gi = np.unique(groups, return_inverse=True)
        gt, n_g = torch.as_tensor(gi, device=DEV), len(ug)
    best, state, bad = np.inf, None, 0
    for ep in range(max_epochs):
        net.train()
        opt.zero_grad()
        eta = net(xt)
        loss = multinomial_nll(eta, yt)
        if penalty and lam > 0:
            r = group_risks(eta, yt, gt, n_g)
            if penalty == "vrex":
                loss = loss + lam * r.var()
            elif penalty == "dro":
                loss = (1 - lam) * loss + lam * torch.logsumexp(r * 10, 0) / 10
        loss.backward()
        opt.step()
        if ep % 5 == 4:
            net.eval()
            with torch.no_grad():
                v = float(multinomial_nll(net(xv), yv))
            if v < best - 1e-4:
                best, bad, state = v, 0, {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 5
                if bad >= patience:
                    break
    net.load_state_dict(state)
    return net.eval()


def predict_flat(net, X):
    with torch.no_grad():
        return probs_from_eta(net(torch.as_tensor(X, device=DEV)).cpu().numpy())


def knn_geo(D: Data, tr, Ytr, te, k=30, prior=0.5, exclude_self=False):
    """Distance-weighted composition of the k nearest training report cells (pure spatial interpolation)."""
    from sklearn.neighbors import BallTree
    A = np.radians(np.c_[D.lat[tr], D.lon[tr]])
    B = np.radians(np.c_[D.lat[te], D.lon[te]])
    tree = BallTree(A, metric="haversine")
    kk = min(k + int(exclude_self), len(tr))
    dist, ind = tree.query(B, k=kk)
    if exclude_self:
        dist, ind = dist[:, 1:], ind[:, 1:]
    dkm = dist * 6371.0
    w = 1.0 / (dkm + 10.0)
    sh = Ytr / Ytr.sum(1, keepdims=True)
    comp = (w[..., None] * sh[ind]).sum(1) / w.sum(1, keepdims=True)
    g = Ytr.sum(0) / Ytr.sum()
    neff = w.sum(1, keepdims=True) / w.max(1, keepdims=True)
    P = (comp * neff + prior * g * 5) / (neff + prior * 5)
    return P / P.sum(1, keepdims=True), dkm[:, 0]


def sk_expand(X, Y):
    r, c = np.nonzero(Y)
    return X[r], c, Y[r, c]


def fit_sklearn(kind, Xtr, Ytr, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    xs, ys, ws = sk_expand(Xtr, Ytr)
    if kind == "HGB":
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                           min_samples_leaf=40, early_stopping=True, validation_fraction=0.15,
                                           n_iter_no_change=30, random_state=seed)
    else:
        m = RandomForestClassifier(n_estimators=400, min_samples_leaf=10, max_features="sqrt", n_jobs=-1, random_state=seed)
    m.fit(xs, ys, sample_weight=ws)
    return m


def predict_sklearn(m, X):
    P = np.full((len(X), 5), 1e-6)
    P[:, m.classes_] = np.clip(m.predict_proba(X), 1e-6, 1)
    return P / P.sum(1, keepdims=True)


def fit_net(kind, D: Data, tr, Ytr, va, Yva, seed):
    """GatedMoE / HiCo through the original HiCo-Fuse trainer."""
    st = D.st
    train, val = Problem(st, D.pos[tr], Ytr), Problem(st, D.pos[va], Yva)
    net, _, _ = fit_hico(kind, st, train, val, seed=seed)
    return net


def predict_net(net, D: Data, te, Yte):
    p = Problem(D.st, D.pos[te], Yte)
    with torch.no_grad():
        return probs_from_eta(p.eta(net).cpu().numpy())
