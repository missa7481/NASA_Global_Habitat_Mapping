#!/usr/bin/env python3
"""RSI iteration 1 diagnosis: where and why do baselines fail?

For every out-of-fold test cell (designs block10, loro, env, zone, spatiotemp):
  per-cell NLL of each model vs geographic distance to training, sampling density, environmental novelty.
Outputs (artifacts/rsi/it1/):
  diag_bins.csv        NLL by distance / novelty bin, per model and design
  diag_partial.csv     joint OLS of the MLP - kNN NLL gap on log distance, novelty, density (standardised)
  diag_spearman.csv    rank correlation of per-cell NLL with each support variable
  diag_region.csv      per-region AUC in-distribution (block10 OOF) vs transfer (LORO)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.common import ART, REGIONS, Data, cell_nll, designs, metrics  # noqa: E402
from rsi.support_ctx import env_space, support_context  # noqa: E402

IT1 = ART / "it1"
MODELS = ["Prior", "kNN-geo", "GLM", "MLP", "MLP+Coords", "HGB", "RF", "GatedMoE", "HiCo"]
DIST_BINS = [0, 10, 50, 200, 1000, 1e5]
NOV_BINS = [0, 1, 1.5, 2.5, 4, 1e9]


def main():
    D = Data()
    DS = designs(D)
    E = env_space(D)
    bins, part, spear = [], [], []
    for dname in ("block10", "loro", "env", "zone", "spatiotemp"):
        f = IT1 / f"oof_{dname}.npz"
        if not f.exists():
            continue
        o = np.load(f, allow_pickle=True)
        ctx = {k: [] for k in ("dist_km", "dens250", "env_nov")}
        for tr, Ytr, te, Yte, label in DS[dname]:
            c = support_context(D, tr, te, E)
            for k in ctx:
                ctx[k].append(c[k])
        ctx = {k: np.concatenate(v) for k, v in ctx.items()}
        Y = o["Y"]
        nll = {m: cell_nll(o[m].astype(np.float64), Y) for m in MODELS}
        w = Y.sum(1)
        db = np.digitize(ctx["dist_km"], DIST_BINS) - 1
        nb = np.digitize(ctx["env_nov"], NOV_BINS) - 1
        for var, b, edges in (("dist_km", db, DIST_BINS), ("env_nov", nb, NOV_BINS)):
            for k in range(len(edges) - 1):
                sel = b == k
                if sel.sum() < 30:
                    continue
                for m in MODELS:
                    mt = metrics(o[m][sel].astype(np.float64), Y[sel])
                    bins.append({"design": dname, "var": var, "bin": f"{edges[k]:g}-{edges[k + 1]:g}", "bin_i": k,
                                 "n_cells": int(sel.sum()), "model": m, "nll": float(np.average(nll[m][sel], weights=w[sel])),
                                 "AUC_mean": mt["AUC_mean"]})
        for m in MODELS:
            row = {"design": dname, "model": m}
            for k, v in ctx.items():
                row[k] = spearmanr(nll[m], v).statistic
            spear.append(row)
        # joint decomposition of the "environment model minus spatial interpolator" gap
        Z = np.c_[np.log1p(ctx["dist_km"]), np.log(ctx["env_nov"] + 1e-3), np.log1p(ctx["dens250"])]
        Z = (Z - Z.mean(0)) / Z.std(0)
        for m in ("MLP", "HGB", "HiCo", "MLP+Coords"):
            gap = nll[m] - nll["kNN-geo"]                                 # < 0 means the covariate model wins
            A = np.c_[np.ones(len(Z)), Z]
            beta, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], gap * np.sqrt(w), rcond=None)
            part.append({"design": dname, "model": m, "mean_gap": float(np.average(gap, weights=w)),
                         "b_log_dist": beta[1], "b_log_envnov": beta[2], "b_log_density": beta[3],
                         "corr_dist_envnov": float(spearmanr(ctx["dist_km"], ctx["env_nov"]).statistic)})
        print(dname, "done", flush=True)
    pd.DataFrame(bins).to_csv(IT1 / "diag_bins.csv", index=False)
    pd.DataFrame(part).to_csv(IT1 / "diag_partial.csv", index=False)
    pd.DataFrame(spear).to_csv(IT1 / "diag_spearman.csv", index=False)

    # per-region: in-distribution (block10 OOF restricted to region) vs leave-region-out transfer
    reg = []
    o = np.load(IT1 / "oof_block10.npz", allow_pickle=True)
    rid = D.region[o["idx"]]
    met = pd.read_csv(IT1 / "metrics.csv")
    for r in REGIONS:
        sel = rid == r
        for m in MODELS:
            a = metrics(o[m][sel].astype(np.float64), o["Y"][sel])
            t = met[(met.design == "loro") & (met.split == r) & (met.model == m)]
            reg.append({"region": r, "model": m, "n_cells": int(sel.sum()), "AUC_block10": a["AUC_mean"], "nll_block10": a["nll"],
                        "AUC_loro": float(t["AUC_mean"].iloc[0]) if len(t) else np.nan,
                        "nll_loro": float(t["nll"].iloc[0]) if len(t) else np.nan})
    pd.DataFrame(reg).to_csv(IT1 / "diag_region.csv", index=False)
    print(pd.DataFrame(part).round(3).to_string())


if __name__ == "__main__":
    main()
