#!/usr/bin/env python3
"""Spatial-block CV comparison of HiCo-Fuse architectures, then full fits + global inference.

  python hicofuse/train.py cv        -> artifacts/hicofuse/cv_metrics.csv, cv_summary.csv
  python hicofuse/train.py full      -> artifacts/hicofuse/global/<model>_eta.npy (+ gates, parent A)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicofuse.models import Net, multinomial_nll, parent_logsumexp  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts/hicofuse"
DEV = "cuda"
CLASSES = ["Ae_albopictus", "Ae_aegypti", "Culex", "Anopheles"]
ARCHS = ["LinearEarly", "MLPEarly", "GatedMoE", "HiCo", "HiCoMoE", "HiCo-climA", "HiCoMoE-climA"]
SINGLES = ["Climate", "Landscape", "Human", "AlphaEarth"]


class Store:
    """Standardised features for complete cells + parent tables (label-free preprocessing)."""

    def __init__(self):
        info = json.loads((ART / "dataset_meta.json").read_text())
        self.info = info
        d = np.load(ART / "dataset.npz")
        self.d = d
        complete = d["complete"]
        self.cell = np.flatnonzero(complete)                       # store row ids of complete cells
        pos = np.full(len(complete), -1, np.int64)
        pos[self.cell] = np.arange(len(self.cell))
        self.pos = pos                                             # store row -> compact position
        X = np.load(ART / "store/X.npy", mmap_mode="r")
        Xc = np.asarray(X[self.cell], dtype=np.float32)
        for j in info["log1p_cols"]:
            Xc[:, j] = np.log1p(np.clip(Xc[:, j], 0, None))
        rng = np.random.default_rng(0)
        samp = Xc[rng.choice(len(Xc), 400_000, replace=False)]
        self.mu, self.sd = samp.mean(0), samp.std(0) + 1e-6
        self.X = (Xc - self.mu) / self.sd
        del Xc
        self.cols = info["modality_cols"]
        pk = d["pkey"][self.cell]
        self.par_keys, self.par = np.unique(pk, return_inverse=True)
        n_par = len(self.par_keys)
        cc = self.cols["Climate"]
        cnt = np.bincount(self.par, minlength=n_par).astype(np.float32)
        pc = np.zeros((n_par, len(cc)), np.float32)
        for k, j in enumerate(cc):
            pc[:, k] = np.bincount(self.par, weights=self.X[:, j], minlength=n_par) / cnt
        pa = np.zeros((n_par, self.X.shape[1]), np.float32)
        for j in range(self.X.shape[1]):
            pa[:, j] = np.bincount(self.par, weights=self.X[:, j], minlength=n_par) / cnt
        self.par_clim, self.par_all, self.par_n = pc, pa, cnt
        self.rc = d["rc"][self.cell]

    def batch(self, p: np.ndarray) -> dict:
        x = self.X[p]
        b = {m: x[:, c] for m, c in self.cols.items()}
        b["all"] = x
        b["ClimateAnomaly"] = b["Climate"] - self.par_clim[self.par[p]]
        b["ctx"] = np.c_[b["Climate"], b["Human"]]
        return {k: torch.as_tensor(np.ascontiguousarray(v), device=DEV) for k, v in b.items()}


class Problem:
    """A set of labelled report cells (+ children for hierarchical models)."""

    def __init__(self, st: Store, obs_pos: np.ndarray, Y: np.ndarray):
        self.obs_pos, self.Y = obs_pos, torch.as_tensor(Y, dtype=torch.float32, device=DEV)
        self.flat_b = st.batch(obs_pos)
        par_ids = np.unique(st.par[obs_pos])
        child = np.flatnonzero(np.isin(st.par, par_ids))
        loc_par = np.searchsorted(par_ids, st.par[child])
        self.child_b = st.batch(child)
        self.child_par = torch.as_tensor(loc_par, device=DEV)
        self.n_par = len(par_ids)
        self.pclim = torch.as_tensor(st.par_clim[par_ids], device=DEV)
        self.pall = torch.as_tensor(st.par_all[par_ids], device=DEV)
        self.logn = torch.as_tensor(np.log(st.par_n[par_ids]), device=DEV)
        where = {p: i for i, p in enumerate(child)}
        self.obs_in_child = torch.as_tensor([where[p] for p in obs_pos], device=DEV)

    def eta(self, net: Net) -> torch.Tensor:
        if net.hierarchical:
            e, _ = net.eta_hier(self.child_b, self.child_par, self.n_par, self.pall if net.macro_all else self.pclim, self.logn)
            return e[self.obs_in_child]
        return net.flat(self.flat_b)


def fit(kind, st, train: Problem, val: Problem, seed, max_epochs=800, patience=60):
    torch.manual_seed(seed)
    dims = {m: len(c) for m, c in st.cols.items()}
    net = Net(kind, dims).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    best, best_state, bad = np.inf, None, 0
    for ep in range(max_epochs):
        net.train()
        opt.zero_grad()
        multinomial_nll(train.eta(net), train.Y).backward()
        opt.step()
        if ep % 5 == 4:
            net.eval()
            with torch.no_grad():
                v = float(multinomial_nll(val.eta(net), val.Y))
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 5
                if bad >= patience:
                    break
    net.load_state_dict(best_state)
    net.eval()
    return net, best, ep + 1


def class_metrics(eta: np.ndarray, Y: np.ndarray) -> dict:
    out = {}
    for s, name in enumerate(CLASSES):
        y = (Y[:, s] > 0).astype(int)
        out[f"AUC_{name}"] = roc_auc_score(y, eta[:, s]) if 0 < y.sum() < len(y) else np.nan
        out[f"AP_{name}"] = average_precision_score(y, eta[:, s]) if y.sum() else np.nan
    return out


def split_inner(st, pos, fold_blocks, seed):
    """Hold out ~15% of training 10-degree blocks for early stopping."""
    rng = np.random.default_rng(seed)
    ub = np.unique(fold_blocks)
    val_b = rng.choice(ub, max(1, int(round(0.15 * len(ub)))), replace=False)
    v = np.isin(fold_blocks, val_b)
    return pos[~v], pos[v], v


def cv(seeds=(0, 1), archs=None, tag="cv", singles=True):
    archs = archs or ARCHS
    st = Store()
    d = st.d
    obs_pos = st.pos[d["obs_idx"]]
    Y = d["Y"]
    fold = d["fold"]
    blocks = (st.rc[obs_pos, 0] // 200).astype(np.int64) * 100 + st.rc[obs_pos, 1] // 200
    rows = []
    t0 = time.time()
    for k in range(5):
        te = fold == k
        test = Problem(st, obs_pos[te], Y[te])
        for seed in seeds:
            trp, vap, vmask = split_inner(st, np.flatnonzero(~te), blocks[~te], 100 * k + seed)
            train = Problem(st, obs_pos[trp], Y[trp])
            val = Problem(st, obs_pos[vap], Y[vap])
            single_eta = {}
            for kind in archs + ([f"Single:{m}" for m in SINGLES] if singles else []):
                net, vbest, ep = fit(kind, st, train, val, seed=1000 * k + seed)
                with torch.no_grad():
                    e = test.eta(net)
                    nll = float(multinomial_nll(e, test.Y))
                e = e.cpu().numpy()
                if kind.startswith("Single:"):
                    single_eta[kind] = e
                rows.append({"fold": k, "seed": seed, "model": kind, "test_nll": nll, "val_nll": vbest, "epochs": ep,
                             "n_test_cells": int(te.sum()), **class_metrics(e, Y[te])})
            if not single_eta:
                print(f"fold {k} seed {seed} done {time.time() - t0:.0f}s", flush=True)
                continue
            # late rank ensemble across single-modality experts (reference-style blending)
            ranks = [pd.DataFrame(v).rank(pct=True).to_numpy() for v in single_eta.values()]
            ens = np.mean(ranks, axis=0)
            rows.append({"fold": k, "seed": seed, "model": "LateRankEnsemble", "test_nll": np.nan, "val_nll": np.nan,
                         "epochs": 0, "n_test_cells": int(te.sum()), **class_metrics(ens, Y[te])})
            print(f"fold {k} seed {seed} done {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ART / f"{tag}_metrics.csv", index=False)
    met = ["test_nll"] + [f"AUC_{c}" for c in CLASSES] + [f"AP_{c}" for c in CLASSES]
    summ = res.groupby("model")[met].mean()
    summ["AUC_mean"] = summ[[f"AUC_{c}" for c in CLASSES]].mean(1)
    summ = summ.sort_values("AUC_mean", ascending=False)
    summ.to_csv(ART / f"{tag}_summary.csv")
    with pd.option_context("display.width", 250):
        print(summ.round(3))


def full(models=("MLPEarly", "GatedMoE", "HiCo", "HiCoMoE")):  # noqa: B006
    st = Store()
    d = st.d
    obs_pos = st.pos[d["obs_idx"]]
    Y = d["Y"]
    blocks = (st.rc[obs_pos, 0] // 200).astype(np.int64) * 100 + st.rc[obs_pos, 1] // 200
    out = ART / "global"
    out.mkdir(exist_ok=True)
    np.save(out / "cell_store_rows.npy", st.cell)
    np.save(out / "rc.npy", st.rc)
    np.save(out / "parent.npy", st.par)
    log = {}
    for kind in models:
        etas, As = [], []
        for seed in (0, 1, 2):
            trp, vap, _ = split_inner(st, np.arange(len(obs_pos)), blocks, 7 + seed)
            net, vbest, ep = fit(kind, st, Problem(st, obs_pos[trp], Y[trp]), Problem(st, obs_pos[vap], Y[vap]), seed=seed)
            etas.append(predict_global(net, st, out, kind, save_parts=(seed == 0)))
            if net.hierarchical:
                As.append(np.load(out / f"{kind}_parentA_seed.npy"))
            log[f"{kind}_seed{seed}"] = {"val_nll": vbest, "epochs": ep}
            print(kind, seed, vbest, ep, flush=True)
        # Ensemble in INTENSITY space (log-mean-exp), never by averaging logs: additivity, and hence
        # child->parent coherence, is preserved only by linear averaging of intensities.
        eta = (np.logaddexp.reduce(np.stack(etas), axis=0) - np.log(len(etas))).astype(np.float32)
        np.save(out / f"{kind}_eta.npy", eta)
        if As:
            A = np.logaddexp.reduce(np.stack(As), axis=0) - np.log(len(As))
            np.save(out / f"{kind}_parentA.npy", A.astype(np.float32))
    (out / "full_fit_log.json").write_text(json.dumps(log, indent=1))


@torch.no_grad()
def predict_global(net, st, out, kind, chunk=400_000, save_parts=False):
    n = len(st.cell)
    if not net.hierarchical:
        eta = np.concatenate([net.flat(st.batch(np.arange(i, min(i + chunk, n)))).cpu().numpy() for i in range(0, n, chunk)])
        if save_parts and kind == "GatedMoE":
            g = np.concatenate([net.f({m: b[m] for m in ("Climate", "Landscape", "Human", "AlphaEarth")}, b["ctx"], True)[1].cpu().numpy()
                                for b in (st.batch(np.arange(i, min(i + chunk, n))) for i in range(0, n, chunk))]).astype(np.float16)
            np.save(out / f"{kind}_gates.npy", g)
        return eta
    n_par = len(st.par_keys)
    A = net.A(torch.as_tensor(st.par_all if net.macro_all else st.par_clim, device=DEV)).cpu().numpy()
    Bs, gates = [], []
    for i in range(0, n, chunk):
        b = st.batch(np.arange(i, min(i + chunk, n)))
        if kind == "HiCoMoE" and save_parts:
            e, g, _ = net.local(b, return_parts=True)
            gates.append(g.cpu().numpy().astype(np.float16))
        else:
            e = net.local(b)
        Bs.append(e.cpu().numpy())
    B = np.concatenate(Bs)
    lse = parent_logsumexp(torch.as_tensor(B, device=DEV), torch.as_tensor(st.par, device=DEV), n_par).cpu().numpy()
    eta = A[st.par] + B - lse[st.par] + np.log(st.par_n)[st.par][:, None]
    np.save(out / f"{kind}_parentA_seed.npy", A.astype(np.float32))
    if save_parts:
        if gates:
            np.save(out / f"{kind}_gates.npy", np.concatenate(gates))
    return eta


if __name__ == "__main__":
    if sys.argv[1] == "cv":
        cv()
    elif sys.argv[1] == "cv-hier":
        cv(archs=["HiCo", "HiCoMoE", "HiCo-climA", "HiCoMoE-climA"], tag="cv_hier", singles=False)
    else:
        full()
