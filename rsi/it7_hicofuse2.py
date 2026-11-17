#!/usr/bin/env python3
"""RSI iteration 7: HiCo-Fuse 2, rebuilt from the iteration-6 diagnosis.

I6-01 showed that HiCo's coherence rule is free (RF+Recon keeps RF accuracy) and that HiCo's learned neural
local and macro terms are what lose under transfer. HiCo-Fuse 2 keeps the hierarchy and replaces both terms:

    eta_s(x) = A_s(P) + tau_s * B_s(x) - log mean_{x' in P} exp(tau_s * B_s(x'))          (exactly coherent)
    A_s(P)   = alpha_s + gamma_s * M_s(P) + w_s . z(P)
    B_s(x)   = log-ratio of a random-forest niche model fused over all four modalities
    M_s(P)   = log mean_{x' in P} exp(B_s(x'))      (the RF-aggregated parent map)
    z(P)     = exogenous species-pool layer at the parent (Kraemer compendium density, 100/500 km, 2 species)

Only 7 x 4 = 28 calibration parameters are learned, with a ridge penalty towards the identity
(alpha = 0, gamma = 1, tau = 1, w = 0, where HiCo-Fuse 2 equals the RF exactly). They are fitted on
spatially out-of-fold RF predictions (5-degree blocks inside the training set, lesson M1-2), never on
in-sample RF output.

Arms: RF | RF+Recon (I6) | HF2-cal (no pool) | HF2 (with pool). Scored at 0.05-degree cells and 0.5-degree
parents (pooled held-out counts). Prediction: HF2 >= RF under LORO/zone (pool carries the regional prior that
I1-03 identified as the binding constraint), ~= RF under block10. Falsified if HF2 loses to RF under LORO.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.neighbors import BallTree, NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.common import ROOT, ART, Data, designs, fit_sklearn, metrics, predict_sklearn  # noqa: E402

import os
BLOCK_DEG = float(os.environ.get("HF2_BLOCK", "5"))
POOL = os.environ.get("HF2_POOL", "kraemer")          # "kraemer" (I7-01..04) or "all" (I7-05: + GBIF genus pool)
OUT = ART / (("it7" if BLOCK_DEG == 5 else f"it7_b{int(BLOCK_DEG)}") + ("" if POOL == "kraemer" else "_pool4"))
OUT.mkdir(parents=True, exist_ok=True)
DESIGNS = ("block10", "loro", "env", "zone")
K_OOF = 4
LAM = 0.01


def eta_of(P):
    P = np.clip(P, 1e-4, 1)
    return np.log(P[:, :4]) - np.log(P[:, 4:5])


def softmax_eta(eta):
    z = np.c_[eta, np.zeros(len(eta))]
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def rf_eta(m, X, bs=200_000):
    return np.concatenate([eta_of(predict_sklearn(m, X[i:i + bs])) for i in range(0, len(X), bs)]) if len(X) else np.zeros((0, 4))


def pool_parent(Y, g, n):
    out = np.zeros((n, Y.shape[1]), np.float32)
    np.add.at(out, g, Y)
    return out


# ----------------------------------------------------------------------------- parent pool layer
def parent_pool(st):
    """z(P): standardised log1p Kraemer compendium cells within 100/500 km of each parent centroid."""
    n = len(st.par_keys)
    lat = 85 - (st.rc[:, 0] + 0.5) * 0.05
    lon = -180 + (st.rc[:, 1] + 0.5) * 0.05
    cnt = np.bincount(st.par, minlength=n)
    plat = np.bincount(st.par, weights=lat, minlength=n) / cnt
    plon = np.bincount(st.par, weights=lon, minlength=n) / cnt
    q = np.radians(np.c_[plat, plon])
    d = st.d
    out = []
    for sp in ("aegypti", "albopictus"):
        rc = d["rc"][d[f"kraemer_{sp}"]]
        ll = np.radians(np.c_[85 - (rc[:, 0] + 0.5) * 0.05, -180 + (rc[:, 1] + 0.5) * 0.05])
        t = BallTree(ll, metric="haversine")
        for km in (100, 500):
            out.append(np.log1p(t.query_radius(q, r=km / 6371.0, count_only=True)))
    if POOL == "all":
        out += gbif_pool(plat, plon)
    Z = np.stack(out, 1).astype(np.float64)
    return (Z - Z.mean(0)) / Z.std(0)


def gbif_pool(plat, plon):
    """GBIF Culicidae 1900-2021 per 5-degree tile (rsi/fetch_gbif_pool.py): log1p total and smoothed genus shares,
    for the parent's own tile and its 3x3 tile neighbourhood."""
    import json
    T = json.loads((ART / "gbif_pool/tiles.json").read_text())
    ti, tj = np.floor(plat / 5).astype(int), np.floor(plon / 5).astype(int)
    keys = ("total", "Aedes", "Culex", "Anopheles")

    def cnt(i, j):
        v = T.get(f"{i},{j}")
        return np.array([v[k] for k in keys], float) if v else np.zeros(4)

    own = np.array([cnt(i, j) for i, j in zip(ti, tj)])
    nb = np.array([sum(cnt(i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1)) for i, j in zip(ti, tj)])
    out = []
    for C in (own, nb):
        out.append(np.log1p(C[:, 0]))
        for g in (1, 2, 3):
            out.append(np.log((C[:, g] + 1) / (C[:, 0] + 4)))
    return out


# ----------------------------------------------------------------------------- the HiCo-Fuse 2 head
class Head:
    """28-parameter coherent head over a fixed local niche term."""

    def __init__(self, dz, pool=True):
        self.pool = pool
        self.th = {"a": torch.zeros(4, dtype=torch.float64), "g": torch.ones(4, dtype=torch.float64),
                   "t": torch.ones(4, dtype=torch.float64), "w": torch.zeros(dz, 4, dtype=torch.float64)}
        self.th0 = {k: v.clone() for k, v in self.th.items()}

    @staticmethod
    def _lme(B, g, n):
        idx = torch.as_tensor(g)[:, None].expand_as(B)
        mx = torch.full((n, 4), -1e30, dtype=B.dtype).scatter_reduce(0, idx, B, "amax", include_self=True)
        se = torch.zeros((n, 4), dtype=B.dtype).index_add(0, torch.as_tensor(g), torch.exp(B - mx[g]))
        return mx + torch.log(se) - torch.log(torch.as_tensor(np.bincount(g, minlength=n), dtype=B.dtype))[:, None]

    def forward(self, Bobs, gobs, Bch, gch, n, Z, rho=None):
        """rho (n,) in [0, 1]: per-parent trust in the head; 0 -> plain RF. Constant within a parent, so still coherent."""
        th = self.th
        Bch, Bobs = torch.as_tensor(Bch, dtype=torch.float64), torch.as_tensor(Bobs, dtype=torch.float64)
        r = torch.ones((n, 1), dtype=torch.float64) if rho is None else torch.as_tensor(rho, dtype=torch.float64)[:, None]
        a = r * th["a"]
        g = 1 + r * (th["g"] - 1)
        t = 1 + r * (th["t"] - 1)
        M = self._lme(Bch, gch, n)
        Lt = self._lme(t[gch] * Bch, gch, n)
        A = a + g * M
        if self.pool:
            A = A + r * (torch.as_tensor(Z, dtype=torch.float64) @ th["w"])
        eta = A[gobs] + t[gobs] * Bobs - Lt[gobs]
        return eta, A

    def fit(self, Bobs, gobs, Bch, gch, n, Z, Y, lam=LAM):
        keys = ["a", "g", "t"] + (["w"] if self.pool else [])
        for k in keys:
            self.th[k].requires_grad_(True)
        Yt = torch.as_tensor(Y, dtype=torch.float64)
        opt = torch.optim.LBFGS([self.th[k] for k in keys], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            eta, _ = self.forward(Bobs, gobs, Bch, gch, n, Z)
            logits = torch.cat([eta, torch.zeros(len(eta), 1, dtype=eta.dtype)], 1)
            nll = -(Yt * torch.log_softmax(logits, 1)).sum() / Yt.sum()
            pen = sum(((self.th[k] - self.th0[k]) ** 2).sum() for k in keys)
            loss = nll + lam * pen
            loss.backward()
            return loss

        opt.step(closure)
        for k in keys:
            self.th[k] = self.th[k].detach()
        return self

    def predict(self, *a, rho=None):
        with torch.no_grad():
            eta, A = self.forward(*a, rho=rho)
        return eta.numpy(), A.numpy()

    def params(self):
        return {k: v.numpy().round(4).tolist() for k, v in self.th.items()}


# ----------------------------------------------------------------------------- per-split pieces
def children_of(st, pars):
    """Store positions of all complete cells in the given parents + local parent index."""
    ch = np.flatnonzero(np.isin(st.par, pars))
    return ch, np.searchsorted(pars, st.par[ch])


def novelty(F, query_pars, ref_pars, k=5):
    """Mean distance from each query parent's features F[p] to its k nearest reference parents."""
    ref = np.unique(ref_pars)
    nn = NearestNeighbors(n_neighbors=min(k, len(ref))).fit(F[ref])
    return nn.kneighbors(F[query_pars])[0].mean(1)


def trust(d, q):
    """Pre-declared rule: full head up to the OOF 90th-percentile novelty q, linear to plain RF at 2q."""
    return np.clip(2 - d / q, 0, 1)


def oof_problem(D, tr, Ytr, si, Zall):
    """Spatially out-of-fold RF log-ratios for training cells and their parents' children."""
    st = D.st
    blk = (np.floor(D.lat[tr] / BLOCK_DEG).astype(int) * 1000 + np.floor(D.lon[tr] / BLOCK_DEG).astype(int))
    ub = np.unique(blk)
    rng = np.random.default_rng(700 + si)
    fold = rng.permutation(len(ub)) % K_OOF
    f = fold[np.searchsorted(ub, blk)]
    par_tr = st.par[D.pos[tr]]
    Bobs = np.zeros((len(tr), 4))
    gobs = np.zeros(len(tr), np.int64)
    Bch, gch, pars_all, off, dnov, dpool = [], [], [], 0, [], []
    for k in range(K_OOF):
        ink = f == k
        dnov.append(novelty(st.par_all, np.unique(par_tr[ink]), par_tr[~ink]))
        dpool.append(novelty(Zall, np.unique(par_tr[ink]), par_tr[~ink]))
        m = fit_sklearn("RF", D.X[tr[~ink]], Ytr[~ink], seed=si * 10 + k)
        Bobs[ink] = rf_eta(m, D.X[tr[ink]])
        pk = np.unique(par_tr[ink])
        ch, g = children_of(st, pk)
        Bch.append(rf_eta(m, st.X[ch]))
        gch.append(g + off)
        gobs[ink] = np.searchsorted(pk, par_tr[ink]) + off
        pars_all.append(pk)
        off += len(pk)
    q90 = (float(np.quantile(np.concatenate(dnov), 0.9)), float(np.quantile(np.concatenate(dpool), 0.9)))
    return Bobs, gobs, np.concatenate(Bch), np.concatenate(gch), off, np.concatenate(pars_all), q90


def main():
    t0 = time.time()
    D = Data()
    st = D.st
    DS = designs(D)
    Zall = parent_pool(st)
    par_obs_all = st.par[D.pos]
    rows, params = [], []
    for dname in DESIGNS:
        for si, (tr, Ytr, te, Yte, label) in enumerate(DS[dname]):
            # ---- fit the 28-parameter heads on spatially out-of-fold RF output
            Bo, go, Bc, gc, n, pk, q90 = oof_problem(D, tr, Ytr, si, Zall)
            heads = {"HF2-cal": Head(4, pool=False).fit(Bo, go, Bc, gc, n, Zall[pk], Ytr),
                     "HF2": Head(Zall.shape[1], pool=True).fit(Bo, go, Bc, gc, n, Zall[pk], Ytr)}
            for h, H in heads.items():
                params.append({"design": dname, "split": label, "model": h, **H.params()})
            # ---- full RF on the training set; local term on the test parents' children
            rf = fit_sklearn("RF", D.X[tr], Ytr, seed=si)
            pte = np.unique(par_obs_all[te])
            ch, gch = children_of(st, pte)
            Bch = rf_eta(rf, st.X[ch])
            gte = np.searchsorted(pte, par_obs_all[te])
            Bte = rf_eta(rf, D.X[te])
            # independent coarse RF for RF+Recon (as I6)
            pall_ids = np.unique(par_obs_all)
            gpa = np.searchsorted(pall_ids, par_obs_all)
            Ytr_par = pool_parent(Ytr, gpa[tr], len(pall_ids))
            keep = np.flatnonzero(Ytr_par.sum(1) > 0)
            rfc = fit_sklearn("RF", st.par_all[pall_ids[keep]], Ytr_par[keep], seed=si)
            A_rfc = eta_of(predict_sklearn(rfc, st.par_all[pte]))
            M = Head._lme(torch.as_tensor(Bch, dtype=torch.float64), gch, len(pte)).numpy()
            Yc = pool_parent(Yte, gte, len(pte))
            res_f = {"RF": softmax_eta(Bte), "RF+Recon": softmax_eta(A_rfc[gte] + Bte - M[gte])}
            res_c = {"RF-agg": softmax_eta(M), "RF+Recon": softmax_eta(A_rfc)}
            for h, H in heads.items():
                e, A = H.predict(Bte, gte, Bch, gch, len(pte), Zall[pte])
                res_f[h], res_c[h] = softmax_eta(e), softmax_eta(A)
                # coherence check on the test parents: mean_children exp(eta) == exp(A)
                ec, _ = H.predict(Bch, gch, Bch, gch, len(pte), Zall[pte])
                err = np.abs(Head._lme(torch.as_tensor(ec), gch, len(pte)).numpy() - A).max()
                assert err < 1e-6, (h, err)
            # ---- HF2-shrink: same HF2 head, pulled back to plain RF in novel parents (rule fixed before the run)
            # (I7-04) pool-space novelty, and the stricter of the two trusts
            rho_c = trust(novelty(st.par_all, pte, par_obs_all[tr]), q90[0])
            rho_p = trust(novelty(Zall, pte, par_obs_all[tr]), q90[1])
            H = heads["HF2"]
            for arm, rho in (("HF2-shrink", rho_c), ("HF2-shrinkP", rho_p), ("HF2-shrink2", np.minimum(rho_c, rho_p))):
                e, A = H.predict(Bte, gte, Bch, gch, len(pte), Zall[pte], rho=rho)
                res_f[arm], res_c[arm] = softmax_eta(e), softmax_eta(A)
                ec, _ = H.predict(Bch, gch, Bch, gch, len(pte), Zall[pte], rho=rho)
                assert np.abs(Head._lme(torch.as_tensor(ec), gch, len(pte)).numpy() - A).max() < 1e-6
                w_te = rho[gte]
                params.append({"design": dname, "split": label, "model": arm, "q90_cov": q90[0], "q90_pool": q90[1],
                               "trust_mean_cells": float(w_te.mean()), "share_full": float((w_te == 1).mean()),
                               "share_rf": float((w_te == 0).mean())})
            for m, P in res_f.items():
                rows.append({"design": dname, "split": label, "model": m, "level": "fine", **metrics(P, Yte)})
            for m, P in res_c.items():
                rows.append({"design": dname, "split": label, "model": m, "level": "coarse", **metrics(P, Yc)})
            print(f"{dname:8s} {label:14s} n_te={len(te)} parents_te={len(pte)}  {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
            pd.DataFrame(params).to_json(OUT / "head_params.json", orient="records", indent=1)
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 250):
        print(df.groupby(["level", "model", "design"])[["AUC_mean", "nll"]].mean().unstack("design").round(3))
    tests = []
    for lev, ref in (("fine", "RF"), ("coarse", "RF-agg")):
        w = df[df.level == lev].pivot_table(index=["design", "split"], columns="model", values=["AUC_mean", "nll"])
        for m in w["nll"].columns:
            if m == ref:
                continue
            for met, better in (("AUC_mean", 1), ("nll", -1)):
                for scope, sub in (("all", w), ("transfer", w.loc[["loro", "zone"]])):
                    d = (sub[met][m] - sub[met][ref]).dropna() * better
                    k = int((d > 0).sum())
                    tests.append({"level": lev, "ref": ref, "model": m, "metric": met, "scope": scope, "wins": k,
                                  "n": len(d), "mean_diff": float(d.mean() * better), "p_sign": binomtest(k, len(d)).pvalue})
    T = pd.DataFrame(tests)
    T.to_csv(OUT / "paired.csv", index=False)
    print(T.round(4).to_string())


if __name__ == "__main__":
    main()
