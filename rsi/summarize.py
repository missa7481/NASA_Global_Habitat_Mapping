#!/usr/bin/env python3
"""Collect RSI results into artifacts/rsi/report/results.json (tables + chart data for the report page)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts/rsi"
OUT = ART / "report"


def r(x, k=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


def tab(df):
    return json.loads(df.to_json(orient="split", double_precision=4))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    m1 = pd.read_csv(ART / "it1/metrics.csv")
    order = ["random5", "block5", "block10", "block20", "loro", "env", "zone", "temporal", "spatiotemp"]
    g = m1.groupby(["model", "design"])
    for met in ("AUC_mean", "nll", "ECE"):
        p = g[met].mean().unstack("design")[[d for d in order if d in m1.design.unique()]]
        res[f"t1_{met}"] = tab(p.round(4))
    # per-split spread for the main metric
    res["t1_auc_sd"] = tab(g["AUC_mean"].std().unstack("design").round(4))
    # rank of each model per design
    res["t1_rank"] = tab(g["AUC_mean"].mean().unstack("design").rank(ascending=False).astype(int))
    if (ART / "it1/diag_bins.csv").exists():
        b = pd.read_csv(ART / "it1/diag_bins.csv")
        res["diag_bins"] = b.round(4).to_dict(orient="records")
        res["diag_partial"] = pd.read_csv(ART / "it1/diag_partial.csv").round(4).to_dict(orient="records")
        res["diag_spearman"] = pd.read_csv(ART / "it1/diag_spearman.csv").round(3).to_dict(orient="records")
        res["diag_region"] = pd.read_csv(ART / "it1/diag_region.csv").round(4).to_dict(orient="records")
    if (ART / "it2/metrics.csv").exists():
        m2 = pd.read_csv(ART / "it2/metrics.csv")
        s = m2.groupby(["strategy", "design"])[["AUC_mean", "nll", "ECE", "inner_nll"]].mean()
        res["t2"] = s.round(4).reset_index().to_dict(orient="records")
        # generalisation of learned weighting: gain over Equal, inner vs outer
        eq = m2[m2.strategy == "Equal"].set_index(["design", "split"])
        gap = []
        for st_, d in m2.groupby("strategy"):
            d = d.set_index(["design", "split"])
            gi = (eq["inner_nll"] - d["inner_nll"]).groupby("design").mean()
            go = (eq["nll"] - d["nll"]).groupby("design").mean()
            for des in go.index:
                gap.append({"strategy": st_, "design": des, "inner_gain": r(gi.get(des, np.nan), 4), "outer_gain": r(go[des], 4)})
        res["t2_gap"] = gap
        res["t2_w"] = pd.read_csv(ART / "it2/stack_weights.csv").groupby("design").mean(numeric_only=True).round(3).reset_index().to_dict(orient="records")
        gw = pd.read_csv(ART / "it2/gate_weights.csv.gz")
        gw["dist_km"] = np.expm1(gw["log_dist"])
        gw["dbin"] = pd.cut(gw["dist_km"], [-1, 10, 50, 200, 1000, 1e5], labels=["<10", "10-50", "50-200", "200-1000", ">1000"])
        res["gate_by_dist"] = gw.groupby(["dbin", "expert"], observed=True)["w"].mean().unstack().round(3).reset_index().astype({"dbin": str}).to_dict(orient="records")
    if (ART / "it3/invariance.csv").exists():
        inv = pd.read_csv(ART / "it3/invariance.csv")
        res["t3_inv"] = inv.groupby(["variant", "design"])[["AUC_mean", "nll"]].mean().round(4).reset_index().to_dict(orient="records")
        res["ae_r2"] = r(pd.read_csv(ART / "it3/ae_r2_from_drivers.csv")["R2_from_drivers"].median())
    if (ART / "it3/transfer_matrix.csv").exists():
        tm = pd.read_csv(ART / "it3/transfer_matrix.csv")
        rows = []
        for f, d in tm.groupby("features"):
            dg = d[d.train == d.test]
            od = d[(d.train != d.test)]
            rows.append({"features": f, "in_region_AUC": r(dg.AUC_mean.mean()), "transfer_AUC": r(od.AUC_mean.mean()),
                         "in_region_nll": r(dg.nll.mean()), "transfer_nll": r(od.nll.mean())})
        res["t3_transfer"] = rows
        res["t3_matrix"] = tm[["features", "train", "test", "AUC_mean", "nll"]].round(4).to_dict(orient="records")
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    print("keys", list(res))


if __name__ == "__main__":
    main()
