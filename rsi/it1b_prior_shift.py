#!/usr/bin/env python3
"""RSI iteration 1b: how much of the transfer loss is class-prior shift?

Under leave-region-out, models beat the class-frequency prior on AUC (~0.70) but not on NLL beyond 1,000 km.
Two re-weightings of the same out-of-fold predictions (no retraining):
  Oracle prior   p'(y|x) ~ p(y|x) * pi_test / pi_train with the TRUE held-out class frequencies (upper bound)
  EM prior       Saerens et al. (2002): pi_test estimated from unlabelled held-out predictions by EM
If the oracle closes most of the NLL gap, the failure is prior (introduction/sampling) shift, not a changed
habitat relation; if not, the conditional p(y|x) itself does not transfer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.common import ART, Data, designs, metrics  # noqa: E402

MODELS = ["GLM", "MLP", "MLP+Coords", "RF", "HGB", "HiCo", "kNN-geo"]


def adjust(P, ratio):
    Q = P * ratio[None]
    return Q / Q.sum(1, keepdims=True)


def em_prior(P, pi_tr, iters=100):
    pi = pi_tr.copy()
    for _ in range(iters):
        Q = adjust(P, pi / pi_tr)
        new = Q.mean(0)
        if np.abs(new - pi).max() < 1e-6:
            break
        pi = new
    return pi


def main():
    D = Data(store=False)
    D.X = np.zeros((len(D.Y), 1))
    rows = []
    for dname in ("block10", "loro", "env", "zone"):
        o = np.load(ART / f"it1/oof_{dname}.npz", allow_pickle=True)
        # rebuild per-split training priors (designs need D.X only for env clusters)
        if dname == "env":
            D2 = Data()
            DS = designs(D2)
        else:
            DS = designs_light(D)
        lab = o["label"]
        for tr, Ytr, te, Yte, label in DS[dname]:
            sel = lab == label
            Y = o["Y"][sel]
            pi_tr = Ytr.sum(0) / Ytr.sum()
            pi_te = Y.sum(0) / Y.sum()
            for m in MODELS:
                P = o[m][sel].astype(np.float64)
                for nm, Q in (("none", P), ("oracle prior", adjust(P, pi_te / pi_tr)),
                              ("EM prior", adjust(P, em_prior(P, pi_tr) / pi_tr))):
                    rows.append({"design": dname, "split": label, "model": m, "correction": nm, **metrics(Q, Y)})
            rows.append({"design": dname, "split": label, "model": "Prior(test, oracle)", "correction": "-",
                         **metrics(np.tile(pi_te, (len(Y), 1)), Y)})
        print(dname, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ART / "it1/prior_shift.csv", index=False)
    with pd.option_context("display.width", 250):
        print(df.groupby(["model", "correction", "design"])["nll"].mean().unstack().round(3))


def designs_light(D):
    # designs() calls env_clusters (needs real X); the other designs only use coordinates / labels
    import rsi.common as c
    orig = c.Data.env_clusters
    c.Data.env_clusters = lambda self, k=5, seed=0: np.zeros(len(self.Y), int)
    try:
        return designs(D)
    finally:
        c.Data.env_clusters = orig


if __name__ == "__main__":
    main()
