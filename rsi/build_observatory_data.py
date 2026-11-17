#!/usr/bin/env python3
"""Aggregate model-development results into ../github_site/data/model_benchmarks.json for the website (Research, Models and Methods pages).

Aggregate metrics only (split means, fold summaries). No coordinates, per-site or per-cell values.
Map images are rendered separately by render_framework_maps.py into ../github_site/assets/hico/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"
SITE = ROOT.parent / "github_site"
OUT = SITE / "data/model_benchmarks.json"

DESIGNS = [
    ("random5", "Random", "Random 5-fold cells: interpolation reference, optimistic"),
    ("block5", "Block 5°", "5° spatial blocks, 5 folds"),
    ("block10", "Block 10°", "10° spatial blocks, 5 folds"),
    ("block20", "Block 20°", "20° spatial blocks, 5 folds"),
    ("loro", "Leave region out", "Train on five world regions, test on the sixth"),
    ("env", "Env. cluster out", "Hold out one of five climate clusters"),
    ("zone", "Climate zone out", "Hold out tropical, arid, temperate or cold"),
    ("temporal", "Temporal", "Train 2022–2024 reports, test 2025"),
    ("spatiotemp", "Space-time", "2025 reports in held-out 10° blocks"),
]
MODEL_LABEL = {"RF": "Random forest", "HGB": "Gradient boosting", "MLP+Coords": "MLP + coordinates", "MLP": "MLP (early fusion)",
               "HiCo": "HiCo hierarchy", "GatedMoE": "Gated mixture of experts", "GLM": "Multinomial GLM", "kNN-geo": "kNN geographic",
               "Prior": "Class prior"}
STRAT_LABEL = {"Equal": "Equal weights", "Theory": "Theory-fixed", "Stack-global": "Stacked (global)", "Stack-region": "Stacked by region",
               "Attn-env": "Attention (environment)", "Attn-support": "Attention (env + support)", "Uncertainty": "Inverse variance",
               "Reliability": "Local reliability"}


# Habitat Representation Lab (a sibling research workspace): a separate Ae. albopictus presence vs
# target-group background project on 1,601 locations. Aggregate numbers transcribed from its reports
# (docs/rsi_report.html, docs/sparse_earth_report.html, docs/cs_representation_report.html, docs/expansion_report.html).
# "use" states what each result changes in the HiCo-Fuse 2 method; tests that do not inform it are listed separately.
LAB = {
    "task": "Ae. albopictus recorded presence vs other-mosquito background · 1,601 locations, 1,292 with held-out assignments · "
            "three 150 km spatial-block folds with 20 km buffers, plus longitude-sector and temperature holdouts · fixed RF and MaxNet probes",
    "fits": "about 600 fixed model fits across five cycles",
    "findings": [
        {"component": "Representation", "verdict": "kept", "title": "Physical context helps a simple model, not the forest",
         "result": "206 physical variables: MaxNet spatial AP +0.042, random forest −0.004",
         "use": "Give the niche term the full physical context; with a forest the gain is already captured."},
        {"component": "Representation", "verdict": "kept", "title": "Simple compression of AlphaEarth is enough",
         "result": "AP: PCA-16 0.831 · raw 64-D 0.821 · learned sparse code 0.809 (learned − raw −0.012, interval −0.028 to +0.010)",
         "use": "Use raw or PCA-compressed embeddings. Sparse dictionaries cut reconstruction error 70% against a random dictionary but do not rank habitat better."},
        {"component": "Representation", "verdict": "rejected", "title": "No general gain from newer self-supervised encoders",
         "result": "Environment + VISReg AP 0.8367 vs environment alone 0.8363 across 27 recipes; expansion VISReg change −0.009",
         "use": "Do not add encoder pretraining to the method without a transfer gain."},
        {"component": "Fusion", "verdict": "rejected", "title": "Adaptive expert gates do not generalise",
         "result": "Equal mixing AP 0.8226 · learned global weights 0.8178 · environment-varying gate 0.8178; the gate failed its pre-declared test in 3 of 3 holdout designs",
         "use": "Independent replication of our weighting (I2) and HiCo (I6) results: learned gates fit the training geography. HiCo-Fuse 2 replaces them with a fixed niche model and a small penalised head."},
        {"component": "Transfer", "verdict": "kept", "title": "AlphaEarth carries transfer to a new continent",
         "result": "Random forest, Americas held out: AP 0.625 → 0.718 with AlphaEarth added",
         "use": "Keep AlphaEarth in the niche term; its value is largest exactly where transfer is hardest."},
        {"component": "Observation", "verdict": "proposed", "title": "Ranking transfers, thresholds do not",
         "result": "All six Europe-held-out models miss every detected site of a 313-site Austrian survey at their training threshold, while keeping ranking signal",
         "use": "Maps must be read as relative rankings; calibrated occurrence needs an explicit observation/detection branch (next step)."},
    ],
    "not_featured": [
        "Input-recovery scores of six unsupervised fusion methods: no habitat endpoint, and correlated views make recovery easy.",
        "Reconstruction gates of sparse autoencoders on their own: reconstruction quality is not habitat prediction.",
        "Masked-JEPA adapter: its training loss rose and its rank collapsed, so it is not a valid test of the method.",
        "Unverified benchmark candidates from literature monitoring (RICapBench, VHR-Bench, PUMA): not admitted.",
    ],
    "next": [
        "Separate ecological state from observation and detection (shared representation, report and survey heads)",
        "Freeze a genuinely unused 2025 validation cohort before any further promotion",
        "Add water, thermal and soil sources one at a time against transfer failures",
    ],
}


def r(x, k=4):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def frame(df):
    return {str(i): {str(c): r(v) for c, v in row.items()} for i, row in df.iterrows()}


def main():
    D = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d"),
         "designs": [{"id": a, "label": b, "desc": c} for a, b, c in DESIGNS]}
    des = [d[0] for d in DESIGNS]

    # ---- iteration 1 holdout matrix
    m1 = pd.read_csv(A / "rsi/it1/metrics.csv")
    order = ["RF", "HGB", "MLP+Coords", "MLP", "HiCo", "GatedMoE", "GLM", "kNN-geo", "Prior"]
    auc = m1.groupby(["model", "design"])["AUC_mean"].mean().unstack().reindex(index=order, columns=des)
    nll = m1.groupby(["model", "design"])["nll"].mean().unstack().reindex(index=order, columns=des)
    D["holdout"] = {"models": [{"id": m, "label": MODEL_LABEL[m]} for m in order], "AUC": frame(auc), "nll": frame(nll),
                    "n_splits": int(m1.groupby("design")["split"].nunique().sum())}
    D["optimism"] = {m: r(auc.loc[m, "random5"] - auc.loc[m, "loro"]) for m in ("RF", "MLP", "kNN-geo")}

    ps = pd.read_csv(A / "rsi/it1/prior_shift.csv")
    pl = ps[ps.design == "loro"].groupby(["model", "correction"])["nll"].mean().unstack()
    D["prior_shift"] = {"rows": [{"model": MODEL_LABEL.get(m, m), "none": r(pl.loc[m, "none"]), "oracle": r(pl.loc[m, "oracle prior"]),
                                  "em": r(pl.loc[m].get("EM prior", np.nan))} for m in ("RF", "HGB", "MLP", "HiCo") if m in pl.index],
                        "heldout_prior": r(ps[(ps.design == "loro") & (ps.model == "Prior(test, oracle)")]["nll"].mean())}

    # ---- iteration 2 weighting
    m2 = pd.read_csv(A / "rsi/it2/metrics.csv")
    eq = m2[m2.strategy == "Equal"].set_index(["design", "split"])
    rows = []
    for s, d in m2.groupby("strategy"):
        if s.startswith("Only:"):
            continue
        d = d.set_index(["design", "split"])
        rows.append({"id": s, "label": STRAT_LABEL.get(s, s),
                     "inner_gain": r((eq.inner_nll - d.inner_nll).xs("loro").mean()) if s != "Equal" else 0.0,
                     "outer_gain": r((eq.nll - d.nll).xs("loro").mean()) if s != "Equal" else 0.0,
                     **{f"AUC_{k}": r(d.AUC_mean.xs(k).mean()) for k in ("block10", "loro", "env") if k in d.index.get_level_values(0)}})
    D["weighting"] = sorted(rows, key=lambda x: -(x["outer_gain"] or 0))

    # ---- iteration 3 causal / invariance / proxy
    inv = pd.read_csv(A / "rsi/it3/invariance.csv").groupby(["variant", "design"])["AUC_mean"].mean().unstack()
    D["invariance"] = frame(inv)
    prox = pd.read_csv(A / "rsi/it3/proxy_decomposition.csv").groupby(["variant", "design"])["AUC_mean"].mean().unstack()
    D["proxy"] = {"AUC": frame(prox), "r2": r(float((A / "rsi/it3/proxy_r2.txt").read_text()))}
    tm = pd.read_csv(A / "rsi/it3/transfer_matrix.csv")
    na_eu = tm[(tm.train == "N America") & (tm.test == "Europe")].set_index("features")["AUC_mean"]
    D["transfer_na_eu"] = {k: r(v) for k, v in na_eu.items()}

    # ---- iteration 4 support decay
    m4 = pd.read_csv(A / "rsi/it4/metrics.csv")
    p4 = m4.pivot_table(index=["design", "split"], columns="strategy", values="AUC_mean")
    D["decay"] = frame(m4.groupby(["strategy", "design"])["AUC_mean"].mean().unstack())

    def sign(a, b):
        d = (p4[a] - p4[b]).dropna()
        w = int((d > 0).sum())
        return {"wins": w, "n": int(len(d)), "p": r(binomtest(w, len(d)).pvalue)}
    D["decay_tests"] = {"vs_gate": sign("Decay [mixed]", "Attn-support [mixed]"), "vs_equal": sign("Decay [mixed]", "Equal")}
    dp = pd.read_csv(A / "rsi/it4/decay_params.csv")
    D["rho_km"] = {reg: r(dp[dp.design == "loro"].groupby("regime")["rho_km"].median()[reg], 0) for reg in ("block", "mixed")}

    # ---- iteration 5 species pool
    m5 = pd.read_csv(A / "rsi/it5_pool.csv").groupby(["model", "features", "design"])["AUC_mean"].mean().unstack()
    D["pool"] = [{"model": m, "features": f, **{k: r(v) for k, v in row.items()}} for (m, f), row in m5.iterrows()]

    # ---- iteration 6 coherence
    m6 = pd.read_csv(A / "rsi/it6/metrics.csv")
    f6 = m6[m6.level == "fine"].groupby(["model", "design"])[["AUC_mean", "nll"]].mean()
    arms = [("RF", "Random forest, scales mapped separately"), ("RF+Recon", "Random forest + HiCo reconciliation"),
            ("HiCo-RF", "HiCo with random-forest local term"), ("HiCo", "HiCo (original)")]
    coh = pd.read_csv(A / "rsi/it6/coherence.csv")
    flip = coh[coh.model.str.startswith("RF")].groupby("design")["dominant_flip"].mean()   # per-design mean, as in the report
    pt = pd.read_csv(A / "rsi/it6/paired_vs_rf.csv").set_index(["level", "model", "metric"])
    D["coherence"] = {"arms": [{"id": a, "label": l, "coherent": a != "RF",
                                **{f"AUC_{d}": r(f6.loc[(a, d), "AUC_mean"]) for d in ("block10", "loro", "env", "zone")},
                                **{f"nll_{d}": r(f6.loc[(a, d), "nll"]) for d in ("block10", "loro", "env", "zone")}} for a, l in arms],
                      "flip_min": r(flip.min()), "flip_max": r(flip.max()),
                      "tests": {m: {"wins": int(pt.loc[("fine", m, "AUC_mean"), "wins_vs_RF"]), "n": int(pt.loc[("fine", m, "AUC_mean"), "n"]),
                                    "p": r(pt.loc[("fine", m, "AUC_mean"), "p_sign"]), "mean_diff": r(pt.loc[("fine", m, "AUC_mean"), "mean_diff"])}
                                for m in ("RF+Recon", "HiCo-RF", "HiCo")}}

    # ---- iteration 7: HiCo-Fuse 2 (rebuilt). Pool-4 run (I7-05) used when present.
    m7 = pd.read_csv(A / "rsi/it7/metrics.csv")
    p4 = A / "rsi/it7_pool4/metrics.csv"
    if p4.exists():
        m7b = pd.read_csv(p4)
        m7b = m7b[m7b.model.isin(["HF2", "HF2-shrink2"])].assign(model=lambda d: d.model + "+pool4")
        m7 = pd.concat([m7, m7b])
    m7 = pd.concat([m7, m6[m6.model == "HiCo"]])
    f7 = m7[m7.level == "fine"]
    g7 = f7.groupby(["model", "design"])[["AUC_mean", "nll"]].mean()
    hf_arms = [("HiCo", "HiCo, original neural terms"), ("RF", "Random forest (flat, the bar)"),
               ("HF2", "HiCo-Fuse 2, Aedes pool"), ("HF2-shrink2", "HiCo-Fuse 2 + novelty trust"),
               ("HF2+pool4", "HiCo-Fuse 2, all-genus pool"), ("HF2-shrink2+pool4", "HiCo-Fuse 2, all-genus pool + trust")]
    hf_arms = [(a, l) for a, l in hf_arms if a in g7.index.get_level_values(0)]
    w7 = f7.pivot_table(index=["design", "split"], columns="model", values=["AUC_mean", "nll"])

    def paired(m, met):
        d = (w7[met][m] - w7[met]["RF"]).dropna() * (1 if met == "AUC_mean" else -1)
        return {"wins": int((d > 0).sum()), "n": int(len(d)), "mean_diff": r(float(d.mean())),
                "p": r(float(binomtest(int((d > 0).sum()), len(d)).pvalue))}
    lo = f7[f7.design == "loro"].pivot_table(index="split", columns="model", values="nll")
    hp = pd.read_json((A / "rsi/it7_pool4/head_params.json") if p4.exists() else (A / "rsi/it7/head_params.json"))
    hh = hp[hp.model == "HF2"]
    D["hf2"] = {"arms": [{"id": a, "label": l, **{f"AUC_{d}": r(g7.loc[(a, d), "AUC_mean"]) for d in ("block10", "env", "loro", "zone")},
                          **{f"nll_{d}": r(g7.loc[(a, d), "nll"]) for d in ("block10", "env", "loro", "zone")},
                          **({"auc_test": paired(a, "AUC_mean"), "nll_test": paired(a, "nll")} if a != "RF" else {})} for a, l in hf_arms],
                "loro_nll": [{"region": reg, **{a: r(lo.loc[reg, a]) for a, _ in hf_arms if a in lo.columns}} for reg in lo.index],
                "gamma_mean": [r(float(np.mean([g[s] for g in hh["g"]]))) for s in range(4)],
                "tau_mean": [r(float(np.mean([t[s] for t in hh["t"]]))) for s in range(4)],
                "n_params": 28}
    ca = pd.read_csv(A / "hicofuse/results/coherence_audit.csv")
    D["hf2"]["coherence_max_abs_log"] = r(float(ca[ca.model == "HF2"]["median_abs_log_ratio"].max()), 6)
    ev7 = pd.read_csv(A / "hicofuse/results/external_validation.csv")
    ev7 = ev7[ev7.model == "HF2 x support (two-part)"]
    D["hf2"]["external"] = {f"{x.dataset} · {x.subset}": r(x.AUC) for x in ev7.itertuples()}
    if p4.exists():
        D["hf2"]["n_params"] = 60
    gm = A / "hicofuse/global/HF2_meta.json"
    if gm.exists():
        D["hf2"]["global"] = {k: v for k, v in json.loads(gm.read_text()).items() if k in ("trust_share_full", "trust_share_rf", "trust_mean", "parents", "cells", "pool")}

    # ---- Habitat Representation Lab (separate Ae. albopictus presence/background project; aggregate results from its reports)
    D["lab"] = LAB

    # ---- HiCo-Fuse architecture CV and external checks
    c = pd.read_csv(A / "hicofuse/cv_summary.csv").set_index("model")
    ch = pd.read_csv(A / "hicofuse/cv_hier_summary.csv").set_index("model")
    c.loc[["HiCo", "HiCoMoE"]] = ch.loc[["HiCo", "HiCoMoE"]].reindex(columns=c.columns).values
    D["hico_cv"] = [{"model": m, **{k: r(row[k]) for k in ("test_nll", "AUC_Ae_albopictus", "AUC_Ae_aegypti", "AUC_Culex", "AUC_Anopheles", "AUC_mean")}}
                    for m, row in c.sort_values("AUC_mean", ascending=False).iterrows()]
    ev = pd.read_csv(A / "hicofuse/results/external_validation.csv")
    ev["col"] = ev.dataset + " · " + ev.subset
    D["external"] = frame(ev.pivot_table(index="model", columns="col", values="AUC"))

    # ---- OA-DSDM
    o = pd.read_csv(A / "oadsdm/Env41_AE64_tuned/summary.csv")
    D["oadsdm"] = [{k: (r(v) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}
                   for row in o.sort_values("NEON_OOF_AUC", ascending=False).to_dict("records")]
    q = pd.read_csv(A / "oadsdm/step1/q1_effort_recoverability.csv")
    D["effort_r2"] = [{k: (r(v) if isinstance(v, float) else v) for k, v in row.items()} for row in q.to_dict("records")]

    # ---- foundation benchmark
    rc = pd.read_csv(A / "results/representation_comparison.csv")
    fm = pd.read_csv(A / "results/fusion_mlp_summary.csv").rename(columns={"representation": "factor"})
    fm["learner"], fm["endpoint"] = "MLP", "presence_background"
    t = pd.concat([rc, fm])
    t = t[t.endpoint == "presence_background"]
    D["foundation"] = [{"representation": row.factor, "learner": row.learner, "AP": r(row.AP_mean), "AP_sd": r(row.AP_fold_SD), "AUC": r(row.AUC_mean)}
                       for row in t.itertuples()]

    # ---- HiCo-Fuse framework report (aggregate tables only; region boxes and per-cell values are not exported)
    rep = json.loads((A / "hicofuse/report/report_data.json").read_text(encoding="utf-8"))
    zm = json.loads((A / "hicofuse/report/zoom_meta.json").read_text(encoding="utf-8"))
    rep["zoom_parents"] = {k: v["parents"] for k, v in zm.items()}
    rep["gate_dominant_share"] = {nm: [r(x) for x in np.load(A / f"hicofuse/results/gate_share_{nm}.npy")] for nm in ("albopictus", "aegypti")}
    D["hico"] = rep

    # ---- research ledger summary
    led = [json.loads(x) for x in (ROOT / "rsi/memory/ledger.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    D["ledger"] = [{"id": e["id"], "level": e["level"].replace(" RSI", ""), "title": e["title"], "status": e["status"],
                    "next": e.get("next", "")} for e in led if not e["id"].startswith("I0")]
    D["ledger_counts"] = pd.Series([e["status"] for e in led]).value_counts().to_dict()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    def clean(o):   # browsers reject NaN/Infinity in JSON
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o
    OUT.write_text(json.dumps(clean(D), indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
