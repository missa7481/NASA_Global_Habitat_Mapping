#!/usr/bin/env python3
"""Step 1 diagnostic: does the presence-background (PB) endpoint reward effort?

Three questions, all answered on the frozen six outer spatial contexts:

Q1  How much of the human-footprint/effort signal is linearly recoverable from
    AlphaEarth64 (spatial-group CV R^2)?
Q2  Do representations that win on the PB endpoint also win on effort-
    controlled survey endpoints (Austrian ovitraps within held-out folds;
    NEON 2026 external diagnostic)?  A representation whose PB rank is high
    but whose survey rank is low is suspected of learning reporting effort.
Q3  Does removing the effort-predictable subspace from AlphaEarth (fitted on
    training folds only) change PB vs survey performance in opposite
    directions?

NEON is never used for any fitting or selection here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oadsdm import data  # noqa: E402


def residualize(A_tr, E_tr, A_te_list):
    """Remove the part of A linearly predictable from E (fit on train only)."""
    sa, se = StandardScaler().fit(A_tr), StandardScaler().fit(E_tr)
    a, e = sa.transform(A_tr), se.transform(E_tr)
    beta, *_ = np.linalg.lstsq(np.c_[e, np.ones(len(e))], a, rcond=None)
    out = [a - np.c_[e, np.ones(len(e))] @ beta]
    for A_te, E_te in A_te_list:
        at, et = sa.transform(A_te), se.transform(E_te)
        out.append(at - np.c_[et, np.ones(len(et))] @ beta)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "oadsdm" / "step1")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    c = data.load()
    m = c.meta
    X = data.transform(c.X)
    XN = data.transform(c.neon_X)
    lon, lat = m["longitude"].to_numpy(), m["latitude"].to_numpy()
    groups = data.grid_groups(lon, lat)

    # ---------------- Q1: effort recoverable from AlphaEarth ----------------
    q1 = []
    for target in data.EFFORT:
        for name, cols in (("AlphaEarth64", c.ae_cols), ("EnvEco37", c.env_eco_cols)):
            y = X[target].to_numpy()
            pred = np.zeros_like(y)
            for tr, te in GroupKFold(5).split(X, y, groups):
                mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
                mdl.fit(X.iloc[tr][cols], y[tr])
                pred[te] = mdl.predict(X.iloc[te][cols])
            q1.append({"effort_covariate": target, "predictors": name, "spatial_cv_R2": r2_score(y, pred)})
    q1 = pd.DataFrame(q1)
    q1.to_csv(args.out / "q1_effort_recoverability.csv", index=False)
    print(q1.pivot(index="effort_covariate", columns="predictors", values="spatial_cv_R2").round(3))

    # ---------------- Q2/Q3: endpoint discordance ----------------
    reps = {
        "Effort4": data.EFFORT,
        "EnvEco37": c.env_eco_cols,
        "Environment41": c.env_cols,
        "AlphaEarth64": c.ae_cols,
        "AlphaEarth64_effortResid": "resid",
        "Env41_AE64": c.env_cols + c.ae_cols,
    }
    rows, neon_scores = [], {k: [] for k in reps}
    for ctx in c.contexts:
        tr, te = data.context_split(c, ctx)
        po_tr = tr & m.block.eq("po").to_numpy()
        po_te = te & m.block.eq("po").to_numpy()
        sv_te = te & m.block.eq("survey").to_numpy()
        y_tr = m.loc[po_tr, "y_po"].to_numpy()
        for name, cols in reps.items():
            if cols == "resid":
                Atr, Ate, An = residualize(
                    X.loc[po_tr, c.ae_cols].to_numpy(), X.loc[po_tr, data.EFFORT].to_numpy(),
                    [(X.loc[te, c.ae_cols].to_numpy(), X.loc[te, data.EFFORT].to_numpy()),
                     (XN[c.ae_cols].to_numpy(), XN[data.EFFORT].to_numpy())])
                Ftr, Fte, Fn = Atr, pd.DataFrame(Ate, index=m.index[te]), An
            else:
                Ftr = X.loc[po_tr, cols].to_numpy()
                Fte = pd.DataFrame(X.loc[te, cols].to_numpy(), index=m.index[te])
                Fn = XN[cols].to_numpy()
            splits = list(GroupKFold(4).split(Ftr, y_tr, groups[po_tr]))
            mdl = make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=splits, scoring="average_precision",
                                     max_iter=5000),
            )
            mdl.fit(Ftr, y_tr)
            s = pd.Series(mdl.decision_function(Fte.to_numpy()), index=Fte.index)
            y_po = m.loc[m.index[po_te], "y_po"]
            y_sv = m.loc[m.index[sv_te], "y_survey"]
            rows.append({
                "context": ctx["id"], "representation": name,
                "PB_AP": average_precision_score(y_po, s[y_po.index]),
                "PB_AUC": roc_auc_score(y_po, s[y_po.index]),
                "Survey_AUC": roc_auc_score(y_sv, s[y_sv.index]) if y_sv.nunique() == 2 else np.nan,
                "n_survey_test": len(y_sv), "survey_pos": int(y_sv.sum()),
            })
            neon_scores[name].append(mdl.decision_function(Fn))
    fold = pd.DataFrame(rows)
    fold.to_csv(args.out / "q2_fold_metrics.csv", index=False)

    ne = c.neon_meta
    prim = ne["independent_gt10km"].astype(bool).to_numpy()
    yN = ne["target"].to_numpy()
    summ = fold.groupby("representation").agg(
        PB_AP=("PB_AP", "mean"), PB_AP_sd=("PB_AP", "std"), PB_AUC=("PB_AUC", "mean"),
        Survey_AUC=("Survey_AUC", "mean"), Survey_AUC_sd=("Survey_AUC", "std"))
    for name, sl in neon_scores.items():
        z = np.mean([(v - v.mean()) / v.std() for v in sl], axis=0)  # rank-equivalent ensemble
        summ.loc[name, "NEON_AUC"] = roc_auc_score(yN[prim], z[prim])
        summ.loc[name, "NEON_AP"] = average_precision_score(yN[prim], z[prim])
        summ.loc[name, "corr_with_log_pop_NEON"] = np.corrcoef(z, XN["worldpop_2024_density"])[0, 1]
    summ = summ.sort_values("PB_AP", ascending=False)
    summ.to_csv(args.out / "q2_summary.csv")
    with pd.option_context("display.width", 200):
        print(summ.round(3))
    (args.out / "README.json").write_text(json.dumps({
        "neon_prevalence": float(yN[prim].mean()), "neon_n": int(prim.sum()),
        "notes": ["Logistic regression, C chosen by 5-degree group CV on PB training rows only.",
                  "Survey_AUC: Austrian sites in the outer test fold, detected (>=1 positive event) vs not.",
                  "NEON: z-scored mean of six context models; diagnostic only, previously inspected."],
    }, indent=2))


if __name__ == "__main__":
    main()
