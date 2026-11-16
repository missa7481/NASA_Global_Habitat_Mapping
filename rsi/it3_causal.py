#!/usr/bin/env python3
"""RSI iteration 3: prediction vs association vs (candidate) causal structure.

Working DAG (see RESEARCH_MEMORY.md):  Climate -> Landscape(vegetation, water) -> Habitat composition Z
  Human settlement -> Z (peridomestic container habitat)  and  Human -> Observation O (who reports)
  AlphaEarth E = f(Climate, Landscape, Human, unobserved surface structure U);  Z -> Reports R <- O
Composition conditional on a report cancels species-agnostic O; species-specific detectability is NOT
identified here (stated assumption, not tested).

(a) Transfer matrix: expert of modality m trained in region i, evaluated in region j.
    A driver whose relation to Z is causal and unconfounded should transfer (stable off-diagonal skill);
    a proxy whose correlation with Z is region-specific should not.
(b) Invariance training under leave-region / environment / zone holdouts:
    ERM vs V-REx (variance of per-region risk) vs soft group-DRO; and causal feature subsets.
(c) AlphaEarth as proxy: R^2 of E from named drivers; does E's residual (E_perp) carry the gain?
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.common import ART, MODS, REGIONS, Data, designs, fit_flat, inner_split, metrics, predict_flat  # noqa: E402

OUT = ART / "it3"
TRAIN_REGIONS = ["N America", "Latin America", "Europe", "Asia", "Oceania"]   # Africa (75 cells) too small to train


def fit_pred(D, X, tr, Ytr, te, sid, seeds=(0, 1), **kw):
    itr, iva = inner_split(D, tr, seed=sid)
    if iva.sum() < 20 or itr.sum() < 50:
        rng = np.random.default_rng(sid)
        iva = rng.random(len(tr)) < 0.15
        itr = ~iva
    return np.mean([predict_flat(fit_flat(X[tr][itr], Ytr[itr], X[tr][iva], Ytr[iva], seed=sid + s, **kw), X[te]) for s in seeds], 0)


def transfer_matrix(D):
    rows = []
    allc = np.arange(len(D.Y))
    feats = {**{m: D.cols[m] for m in MODS}, "All": sum((D.cols[m] for m in MODS), [])}
    for name, cols in feats.items():
        X = D.X[:, cols]
        for i, ri in enumerate(TRAIN_REGIONS):
            tri = allc[D.region == ri]
            # in-region skill: 3-fold 10-degree block CV inside region i
            b = (D.rc[tri, 0] // 200).astype(np.int64) * 100 + D.rc[tri, 1] // 200
            ub = np.unique(b)
            fb = dict(zip(ub, np.random.default_rng(i).permutation(len(ub)) % 3))
            f = np.array([fb[x] for x in b])
            P = np.zeros((len(tri), 5))
            for k in range(3):
                P[f == k] = fit_pred(D, X, tri[f != k], D.Y[tri[f != k]], tri[f == k], sid=100 * i + k)
            rows.append({"features": name, "train": ri, "test": ri, **metrics(P, D.Y[tri])})
            for rj in REGIONS:
                if rj == ri:
                    continue
                te = allc[D.region == rj]
                P = fit_pred(D, X, tri, D.Y[tri], te, sid=100 * i + 50)
                rows.append({"features": name, "train": ri, "test": rj, **metrics(P, D.Y[te])})
        print("transfer", name, flush=True)
    return pd.DataFrame(rows)


def invariance(D, DS):
    rows = []
    allf = sum((D.cols[m] for m in MODS), [])
    sets = {"Drivers (Clim+Land)": D.cols["Climate"] + D.cols["Landscape"],
            "Drivers+Human": D.cols["Climate"] + D.cols["Landscape"] + D.cols["Human"],
            "All (+AlphaEarth)": allf}
    # AlphaEarth residualised on the named drivers (linear, fit on all report cells: label-free)
    drv = D.X[:, D.cols["Climate"] + D.cols["Landscape"] + D.cols["Human"]]
    AE = D.X[:, D.cols["AlphaEarth"]]
    AEhat = cross_val_predict(Ridge(1.0), drv, AE, cv=5)
    r2 = 1 - ((AE - AEhat) ** 2).sum(0) / ((AE - AE.mean(0)) ** 2).sum(0)
    AEperp = (AE - AEhat).astype(np.float32)
    Xperp = np.c_[drv, AEperp].astype(np.float32)
    variants = [("ERM", sets["All (+AlphaEarth)"], {}), ("V-REx l=1", allf, {"penalty": "vrex", "lam": 1.0}),
                ("V-REx l=10", allf, {"penalty": "vrex", "lam": 10.0}), ("GroupDRO", allf, {"penalty": "dro", "lam": 0.5}),
                ("Drivers (Clim+Land)", sets["Drivers (Clim+Land)"], {}), ("Drivers+Human", sets["Drivers+Human"], {}),
                ("Drivers+Human+AE_perp", None, {})]
    t0 = time.time()
    for dname in ("block10", "loro", "env", "zone"):
        for si, (tr, Ytr, te, Yte, label) in enumerate(DS[dname]):
            groups = D.region[tr] if dname != "env" else D.region[tr]
            for vname, cols, kw in variants:
                X = Xperp if cols is None else D.X[:, cols]
                if "penalty" in kw:
                    # group risks defined over training regions (env/zone holdouts: regions; LORO: remaining regions)
                    itr, iva = inner_split(D, tr, seed=900 + si)
                    ps = []
                    for s in (0, 1):
                        net = fit_flat(X[tr][itr], Ytr[itr], X[tr][iva], Ytr[iva], seed=900 + si + s, groups=groups[itr], **kw)
                        ps.append(predict_flat(net, X[te]))
                    P = np.mean(ps, 0)
                else:
                    P = fit_pred(D, X, tr, Ytr, te, sid=900 + si)
                rows.append({"design": dname, "split": label, "variant": vname, **metrics(P, Yte)})
            print(dname, label, f"{time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows), pd.DataFrame({"AE_dim": [f"A{i:02d}" for i in range(64)], "R2_from_drivers": r2})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = Data()
    DS = designs(D)
    tm = transfer_matrix(D)
    tm.to_csv(OUT / "transfer_matrix.csv", index=False)
    inv, r2 = invariance(D, DS)
    inv.to_csv(OUT / "invariance.csv", index=False)
    r2.to_csv(OUT / "ae_r2_from_drivers.csv", index=False)
    with pd.option_context("display.width", 250):
        print(inv.groupby(["variant", "design"])["AUC_mean"].mean().unstack().round(3))
        print("AE R2 from drivers: median %.2f" % r2.R2_from_drivers.median())


if __name__ == "__main__":
    main()
