#!/usr/bin/env python3
"""RSI iteration 5: an exogenous regional species-pool covariate.

Evidence (I1-03): under leave-region-out, re-weighting predictions to the TRUE held-out class frequencies
brings RF log-loss from 1.302 to 1.070, close to the held-out prior itself (1.057). The non-transferable part is
the regional class prior (which species have arrived), not the local habitat relation.
Hypothesis H5: a coarse, independently sourced species-pool signal restores transfer.
  Pool covariates = log1p(number of Kraemer et al. 2015 compendium cells within 100 km and 500 km) for
  Ae. aegypti and Ae. albopictus (literature records 1960-2014; never iNaturalist).
Prediction: large gain under LORO / zone holdouts, little under 10-degree blocks (where nearby training data
already carry the pool). Falsified if LORO gain <= block gain.
Caveat: maps that use these covariates can no longer be validated against Kraemer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.common import ROOT, ART, Data, designs, fit_flat, fit_sklearn, inner_split, metrics, predict_flat, predict_sklearn  # noqa: E402


def pool_features(D):
    d = np.load(ROOT / "artifacts/hicofuse/dataset.npz")
    q = np.radians(np.c_[D.lat, D.lon])
    out = []
    for sp in ("aegypti", "albopictus"):
        rc = d["rc"][d[f"kraemer_{sp}"]]
        ll = np.radians(np.c_[85 - (rc[:, 0] + 0.5) * 0.05, -180 + (rc[:, 1] + 0.5) * 0.05])
        t = BallTree(ll, metric="haversine")
        for km in (100, 500):
            out.append(np.log1p(t.query_radius(q, r=km / 6371.0, count_only=True)))
    F = np.stack(out, 1).astype(np.float32)
    return (F - F.mean(0)) / F.std(0)


def main():
    D = Data()
    DS = designs(D)
    Pf = pool_features(D)
    sets = {"base": D.X, "+pool": np.c_[D.X, Pf]}
    rows = []
    for dname in ("block10", "loro", "zone", "env"):
        for si, (tr, Ytr, te, Yte, label) in enumerate(DS[dname]):
            itr, iva = inner_split(D, tr, seed=300 + si)
            for fs, X in sets.items():
                P = predict_sklearn(fit_sklearn("RF", X[tr], Ytr, seed=si), X[te])
                rows.append({"design": dname, "split": label, "model": "RF", "features": fs, **metrics(P, Yte)})
                P = np.mean([predict_flat(fit_flat(X[tr][itr], Ytr[itr], X[tr][iva], Ytr[iva], seed=300 + si + s), X[te]) for s in (0, 1)], 0)
                rows.append({"design": dname, "split": label, "model": "MLP", "features": fs, **metrics(P, Yte)})
            print(dname, label, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ART / "it5_pool.csv", index=False)
    with pd.option_context("display.width", 250):
        print(df.groupby(["model", "features", "design"])[["AUC_mean", "nll"]].mean().unstack("design").round(3))
        print(df[df.design == "loro"].pivot_table(index=["model", "features"], columns="split", values="AUC_mean").round(3))


if __name__ == "__main__":
    main()
