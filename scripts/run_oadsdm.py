#!/usr/bin/env python3
"""Run the frozen OA-DSDM protocol (oadsdm/PROTOCOL.md).

Outputs (artifacts/oadsdm/<tag>/):
  fold_metrics.csv     per context x model: PB AP/AUC, Austria survey AUC, NEON-2026 fold AUC/AP
  neon_oof.csv         pooled out-of-fold NEON-2026 scores per seed partition and model
  summary.csv          primary and secondary endpoints
  paired_bootstrap.csv siteID-cluster bootstrap of NEON AUC/AP differences vs. reference model
  params.csv           learned observation parameters (sanity checks)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oadsdm import data  # noqa: E402
from oadsdm.model import Config, evaluate_objective, fit, predict_eta  # noqa: E402

L2_GRID = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

MODELS = {
    "PB": Config("PB", use_survey=False, bias_head=False),
    "Pooled": Config("Pooled", pooled=True, bias_head=False),
    "SurveyOnly": Config("SurveyOnly", use_po=False, bias_head=False),
    "ISDM-noBias": Config("ISDM-noBias", bias_head=False),
    "OA-DSDM": Config("OA-DSDM"),
    "PB-linear": Config("PB-linear", use_survey=False, bias_head=False, linear=True),
    "OA-ISDM-linear": Config("OA-ISDM-linear", linear=True),
}


class OrthScaler:
    """Standardise; optionally replace h by [h_perp, e] with h_perp = h minus its linear
    projection on the effort columns (all fitted on training rows only)."""

    def __init__(self, cols, orth):
        self.cols, self.orth = cols, orth
        self.eff = [cols.index(e) for e in data.EFFORT] if orth else []
        self.keep = [j for j in range(len(cols)) if j not in self.eff]

    def fit(self, H):
        self.sc = StandardScaler().fit(H)
        if self.orth:
            Z = self.sc.transform(H)
            A = np.c_[Z[:, self.eff], np.ones(len(Z))]
            self.beta, *_ = np.linalg.lstsq(A, Z[:, self.keep], rcond=None)
        return self

    def transform(self, H):
        Z = self.sc.transform(H)
        if not self.orth:
            return Z
        E = Z[:, self.eff]
        return np.c_[Z[:, self.keep] - np.c_[E, np.ones(len(E))] @ self.beta, E]

    @property
    def width(self):
        return len(self.cols)


def safe_auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def safe_ap(y, s):
    return average_precision_score(y, s) if len(np.unique(y)) == 2 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="Env41_AE64", choices=["Env41_AE64", "Environment41", "AlphaEarth64", "EnvEco37_AE64", "Env41_AE64x3"])
    ap.add_argument("--orth", action="store_true", help="effort-orthogonal representation [h_perp, e] for all models")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-sklearn", action="store_true")
    ap.add_argument("--tune", action="store_true", help="select L2 per model/context by inner-fold objective")
    args = ap.parse_args()
    tag = args.tag or args.rep
    out = Path(__file__).resolve().parents[1] / "artifacts" / "oadsdm" / tag
    out.mkdir(parents=True, exist_ok=True)

    c = data.load()
    m = c.meta
    rep_cols = {
        "Env41_AE64": c.env_cols + c.ae_cols,
        "Environment41": c.env_cols,
        "AlphaEarth64": c.ae_cols,
        "EnvEco37_AE64": c.env_eco_cols + c.ae_cols,
        "Env41_AE64x3": c.env_cols + c.ae_cols + [f"{a}_r{r}" for r in (100, 2000) for a in c.ae_cols],
    }[args.rep]
    X = data.transform(c.X)
    n25_sites, n25_X, n25_ev = data.load_neon2025()
    n25_X = data.transform(n25_X)
    n26 = c.neon_meta[c.neon_meta["independent_gt10km"].astype(bool)]
    n26_X = data.transform(c.neon_X.loc[n26.index])
    if args.rep == "Env41_AE64x3":
        ms = pd.read_pickle(Path(__file__).resolve().parents[1] / "artifacts" / "oadsdm" / "multiscale" / "alphaearth_multiscale.pkl")
        ms = ms[[col for col in rep_cols if col in ms.columns]]
        X, n25_X, n26_X = (fr.join(ms, how="left") for fr in (X, n25_X, n26_X))
        for name, fr in (("dev", X), ("n25", n25_X), ("n26", n26_X)):
            if fr[rep_cols].isna().any().any():
                raise ValueError(f"multiscale features missing for {name}")
    if args.orth:
        for k in MODELS:
            MODELS[k] = replace(MODELS[k], orth_e=len(data.EFFORT))

    clusters = sorted(set(n25_sites["siteID"]) | set(n26["siteID"]))
    pos25 = n25_sites.groupby("siteID")["target"].max()
    pos26 = n26.groupby("siteID")["target"].max()
    cpos = {k: bool(pos25.get(k, 0) or pos26.get(k, 0)) for k in clusters}
    partitions = {s: data.neon_cluster_folds(cpos, s) for s in (42, 202)}

    # Austria: one aggregated binomial row per site; NEON: one row per trap event.
    at_mask = m["block"].eq("survey").to_numpy()
    at_logeff = np.log(m["trap_days"].to_numpy() / np.maximum(m["n_events"].to_numpy(), 1))
    AT_LE0 = float(np.nanmean(at_logeff[at_mask]))
    N_LE0 = float(np.log(n25_ev["trap_hours"]).mean())

    fold_rows, oof, params, tuning = [], [], [], []
    t0 = time.time()
    for ctx in c.contexts:
        seed, f = int(ctx["seed"]), int(ctx["fold"])
        part = partitions[seed]
        tr, te = data.context_split(c, ctx)
        po_tr = tr & m["block"].eq("po").to_numpy()
        po_te = te & m["block"].eq("po").to_numpy()
        at_tr, at_te = tr & at_mask, te & at_mask
        n25_tr = n25_sites["siteID"].map(part).ne(f).to_numpy()
        n26_te = n26["siteID"].map(part).eq(f).to_numpy()

        # ---- scaling fitted on training rows only ----
        def make_batch(po_m, at_m, n25_m, sh, se):
            def Hf(frame):
                if len(frame) == 0:
                    return np.zeros((0, len(rep_cols)), np.float32)
                return sh.transform(frame[rep_cols].to_numpy()).astype(np.float32)
            at_idx = np.flatnonzero(at_m)
            n25_ids = n25_sites.index[n25_m]
            sv_h = np.vstack([Hf(X.iloc[at_idx]), Hf(n25_X.loc[n25_ids])])
            sv_src = np.r_[np.zeros(len(at_idx)), np.ones(len(n25_ids))].astype(np.int64)
            sv_det = np.r_[m["k_events"].to_numpy()[at_idx] > 0, n25_sites.loc[n25_ids, "target"].to_numpy() > 0]
            ev_n25 = n25_ev[n25_ev["sample_id"].isin(n25_ids)]
            pos_of = {sid: i + len(at_idx) for i, sid in enumerate(n25_ids)}
            ev_site = np.r_[np.arange(len(at_idx)), ev_n25["sample_id"].map(pos_of).to_numpy()].astype(np.int64)
            ev_logeff = np.r_[at_logeff[at_idx] - AT_LE0, np.log(ev_n25["trap_hours"].to_numpy()) - N_LE0].astype(np.float32)
            ev_pos = np.r_[m["k_events"].to_numpy()[at_idx], ev_n25["target"].to_numpy()].astype(np.float32)
            ev_neg = np.r_[m["n_events"].to_numpy()[at_idx] - m["k_events"].to_numpy()[at_idx],
                           1 - ev_n25["target"].to_numpy()].astype(np.float32)
            return dict(
                po_h=Hf(X.loc[po_m]), po_e=se.transform(X.loc[po_m, data.EFFORT].to_numpy()).astype(np.float32),
                po_y=m.loc[po_m, "y_po"].to_numpy().astype(np.float32),
                sv_h=sv_h, sv_src=sv_src, sv_det=sv_det, ev_site=ev_site, ev_logeff=ev_logeff,
                ev_pos=ev_pos, ev_neg=ev_neg, n_survey=2,
            )

        def scalers(po_m, at_m, n25_m):
            sh = OrthScaler(rep_cols, args.orth).fit(np.vstack([X.loc[po_m | at_m, rep_cols].to_numpy(),
                                                                n25_X.loc[n25_m, rep_cols].to_numpy()]))
            se = StandardScaler().fit(X.loc[po_m, data.EFFORT].to_numpy())
            return sh, se

        sh, se = scalers(po_tr, at_tr, n25_tr)
        H = lambda frame: sh.transform(frame[rep_cols].to_numpy()).astype(np.float32)  # noqa: E731
        batch = make_batch(po_tr, at_tr, n25_tr, sh, se)

        # ---- inner folds (training data only) for L2 selection ----
        inner = []
        if args.tune:
            tr_cl = sorted(set(n25_sites.loc[n25_tr, "siteID"]))
            ipart = data.neon_cluster_folds({k: bool(pos25.get(k, 0)) for k in tr_cl}, seed + 7)
            for j, isp in enumerate(ctx["inner"]):
                itr, ite = m.index.isin(isp["train_ids"]), m.index.isin(isp["test_ids"])
                ntr = n25_tr & n25_sites["siteID"].map(ipart).ne(j).to_numpy()
                nte = n25_tr & n25_sites["siteID"].map(ipart).eq(j).to_numpy()
                ish, ise = scalers(po_tr & itr, at_tr & itr, ntr)
                inner.append((make_batch(po_tr & itr, at_tr & itr, ntr, ish, ise),
                              make_batch(po_tr & ite, at_tr & ite, nte, ish, ise)))

        H_po_te, H_at_te, H_n26 = H(X.loc[po_te]), H(X.loc[at_te]), H(n26_X.loc[n26.index[n26_te]])
        y_po_te = m.loc[po_te, "y_po"].to_numpy()
        y_at_te = m.loc[at_te, "y_survey"].to_numpy()
        y_n26 = n26.loc[n26_te, "target"].to_numpy()

        scores = {}
        for name in args.models:
            cfg = MODELS[name]
            if args.tune:
                inner_loss = {}
                for lam in L2_GRID:
                    cl = replace(cfg, l2=lam)
                    inner_loss[lam] = float(np.mean([evaluate_objective(fit(cl, btr, seed=1), cl, bte)
                                                     for btr, bte in inner]))
                lam_star = min(inner_loss, key=inner_loss.get)
                cfg = replace(cfg, l2=lam_star)
                tuning.append({"context": ctx["id"], "model": name, "l2_selected": lam_star,
                               **{f"inner_nll_l2={k:g}": v for k, v in inner_loss.items()}})
            s_po, s_at, s_n = [], [], []
            for k in range(args.seeds):
                mdl = fit(cfg, batch, seed=1000 * seed + 10 * f + k)
                off_n = float(mdl.alpha[1]) if cfg.use_survey else float(mdl.c_po)
                s_po.append(predict_eta(mdl, H_po_te))
                s_at.append(predict_eta(mdl, H_at_te))
                s_n.append(predict_eta(mdl, H_n26) + off_n)
                params.append({"context": ctx["id"], "model": name, "member": k,
                               "c_po": float(mdl.c_po), "alpha_AT": float(mdl.alpha[0]), "alpha_NEON": float(mdl.alpha[1]),
                               "p_event_AT": float(1 / (1 + np.exp(-float(mdl.delta[0])))),
                               "p_event_NEON": float(1 / (1 + np.exp(-float(mdl.delta[1])))),
                               "kappa_AT": float(mdl.kappa[0]), "kappa_NEON": float(mdl.kappa[1])})
            scores[name] = tuple(np.mean(v, axis=0) for v in (s_po, s_at, s_n))

        if not args.no_sklearn:  # frozen-style PB GLM anchor
            Ftr = X.loc[po_tr, rep_cols].to_numpy()
            yy = m.loc[po_tr, "y_po"].to_numpy()
            g = data.grid_groups(m.loc[po_tr, "longitude"], m.loc[po_tr, "latitude"])
            sc = StandardScaler().fit(Ftr)
            glm = LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=list(GroupKFold(4).split(Ftr, yy, g)),
                                       scoring="average_precision", max_iter=5000).fit(sc.transform(Ftr), yy)
            sf = lambda Z: glm.decision_function(sc.transform(Z))  # noqa: E731
            scores["PB-GLM-sklearn"] = (sf(X.loc[po_te, rep_cols].to_numpy()), sf(X.loc[at_te, rep_cols].to_numpy()),
                                        sf(n26_X.loc[n26.index[n26_te], rep_cols].to_numpy()))

        for name, (a, b, cc) in scores.items():
            fold_rows.append({"context": ctx["id"], "model": name,
                              "PB_AP": safe_ap(y_po_te, a), "PB_AUC": safe_auc(y_po_te, a),
                              "AT_AUC": safe_auc(y_at_te, b), "NEON_fold_AUC": safe_auc(y_n26, cc),
                              "NEON_fold_AP": safe_ap(y_n26, cc), "n_neon_test": len(y_n26), "neon_pos": int(y_n26.sum())})
            oof.append(pd.DataFrame({"sample_id": n26.index[n26_te], "siteID": n26.loc[n26_te, "siteID"].to_numpy(),
                                     "partition_seed": seed, "fold": f, "model": name, "score": cc, "target": y_n26}))
        print(f"{ctx['id']} done ({time.time() - t0:.0f}s)", flush=True)

    fold = pd.DataFrame(fold_rows)
    fold.to_csv(out / "fold_metrics.csv", index=False)
    oof = pd.concat(oof, ignore_index=True)
    oof.to_csv(out / "neon_oof.csv", index=False)
    pd.DataFrame(params).to_csv(out / "params.csv", index=False)
    if tuning:
        pd.DataFrame(tuning).to_csv(out / "tuning.csv", index=False)

    # ---- primary endpoint: pooled OOF per partition seed, then mean ----
    summ = fold.groupby("model").agg(PB_AP=("PB_AP", "mean"), PB_AP_sd=("PB_AP", "std"),
                                     AT_AUC=("AT_AUC", "mean"), AT_AUC_sd=("AT_AUC", "std"),
                                     NEON_fold_AUC=("NEON_fold_AUC", "mean"))
    for name, g in oof.groupby("model"):
        aucs = [roc_auc_score(d.target, d.score) for _, d in g.groupby("partition_seed")]
        aps = [average_precision_score(d.target, d.score) for _, d in g.groupby("partition_seed")]
        summ.loc[name, "NEON_OOF_AUC"] = np.mean(aucs)
        summ.loc[name, "NEON_OOF_AP"] = np.mean(aps)
    summ = summ.sort_values("NEON_OOF_AUC", ascending=False)
    summ.to_csv(out / "summary.csv")

    # ---- paired siteID-cluster bootstrap vs PB ----
    rng = np.random.default_rng(20261114)
    ref = "PB"
    boot = []
    if ref in oof["model"].unique():
        wide = oof.pivot_table(index=["partition_seed", "sample_id", "siteID", "target"], columns="model", values="score").reset_index()
        for pseed, w in wide.groupby("partition_seed"):
            cl = w["siteID"].unique()
            groups = {k: np.flatnonzero(w["siteID"].to_numpy() == k) for k in cl}
            draws = {mname: {"dAUC": [], "dAP": []} for mname in oof["model"].unique() if mname != ref}
            for _ in range(1000):
                idx = np.concatenate([groups[k] for k in rng.choice(cl, len(cl), replace=True)])
                yb = w["target"].to_numpy()[idx]
                if yb.min() == yb.max():
                    continue
                base_auc = roc_auc_score(yb, w[ref].to_numpy()[idx])
                base_ap = average_precision_score(yb, w[ref].to_numpy()[idx])
                for mname in draws:
                    sb = w[mname].to_numpy()[idx]
                    draws[mname]["dAUC"].append(roc_auc_score(yb, sb) - base_auc)
                    draws[mname]["dAP"].append(average_precision_score(yb, sb) - base_ap)
            for mname, d in draws.items():
                for metric, vals in d.items():
                    v = np.asarray(vals)
                    boot.append({"partition_seed": pseed, "model": mname, "reference": ref, "metric": metric,
                                 "mean": v.mean(), "ci_low": np.quantile(v, 0.025), "ci_high": np.quantile(v, 0.975),
                                 "P_gt_0": (v > 0).mean(), "draws": len(v)})
    pd.DataFrame(boot).to_csv(out / "paired_bootstrap.csv", index=False)
    (out / "run.json").write_text(json.dumps({"rep": args.rep, "n_features": len(rep_cols), "seeds": args.seeds,
                                              "models": args.models, "neon_clusters": len(clusters),
                                              "neon_partitions": {str(k): v for k, v in partitions.items()},
                                              "runtime_s": time.time() - t0}, indent=1))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(summ.round(3))
        if boot:
            print(pd.DataFrame(boot).round(3).to_string())


if __name__ == "__main__":
    main()
