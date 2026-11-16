#!/usr/bin/env python3
"""Semi-synthetic identifiability benchmark for OA-DSDM.

Real locations, real covariates (Environment41 + AlphaEarth64), real survey
designs (Austrian visit counts and trap-days, NEON 2025 events and trap-hours);
only the labels are simulated, so the true suitability eta* is known.

Generative model (per simulation replicate r):
    eta*(x) = s * standardize( MLP_r(h(x)) + beta_u * u(x) )        true habitat log-odds
    u(x)    = standardized human-footprint index (mean z of Effort4)
    PO:     y ~ Bern(sigmoid(eta* + gamma * u + c_po))               reporting bias via effort
    survey: z ~ Bern(sigmoid(eta* + alpha_s)),  y_j ~ Bern(z * p_sj(effort_j))

Factors: gamma (confounding) x beta_u (urbanness is real habitat) x survey design.
Metrics on held-out sites of the frozen seed-42 contexts:
    rho_all     Spearman(eta_hat, eta*) over all held-out PO + survey sites
    rho_urban   same, restricted to held-out sites with u > 0 (where bias acts)
    leak        corr(eta_hat_std - eta*_std, u) on held-out sites (0 = no effort leakage)
    AUC_z       AUC of eta_hat vs simulated occupancy z at held-out survey sites
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oadsdm import data  # noqa: E402
from oadsdm.model import Config, fit, predict_eta  # noqa: E402

MODELS = {
    "PB": Config("PB", use_survey=False, bias_head=False),
    "Pooled": Config("Pooled", pooled=True, bias_head=False),
    "ISDM-noBias": Config("ISDM-noBias", bias_head=False),
    "OA-DSDM": Config("OA-DSDM"),
    "OA-ISDM-linear": Config("OA-ISDM-linear", linear=True),
    "OA-DSDM-orth": Config("OA-DSDM-orth", orth_e=4),
}
ORTH = {"OA-DSDM-orth"}


def orth_rep(H_tr, E_tr, H_list, E_list, eff_idx):
    """[h_perp, e]: remove the linearly effort-predictable part of h (fit on train), append e."""
    keep = [j for j in range(H_tr.shape[1]) if j not in eff_idx]
    A = np.c_[E_tr, np.ones(len(E_tr))]
    beta, *_ = np.linalg.lstsq(A, H_tr[:, keep], rcond=None)
    out = []
    for H, E in zip(H_list, E_list):
        hp = H[:, keep] - np.c_[E, np.ones(len(E))] @ beta
        out.append(np.c_[hp, E].astype(np.float32))
    return out


def zs(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--members", type=int, default=2)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--gammas", nargs="*", type=float, default=[0.0, 1.0, 2.0])
    ap.add_argument("--betas", nargs="*", type=float, default=[0.0, 1.0])
    ap.add_argument("--designs", nargs="*", default=["AT", "NEON", "AT+NEON"])
    ap.add_argument("--truth", choices=["smooth", "nonlinear"], default="smooth")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "oadsdm" / "semisynthetic")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    c = data.load()
    m = c.meta
    cols = c.env_cols + c.ae_cols
    X = data.transform(c.X)
    s25, X25, ev25 = data.load_neon2025()
    X25 = data.transform(X25)

    # One site table: PO sites, Austria survey sites, NEON 2025 survey sites.
    po = m.index[m.block.eq("po")]
    at = m.index[m.block.eq("survey")]
    H_all = np.vstack([X.loc[po, cols], X.loc[at, cols], X25[cols]])
    E_all = np.vstack([X.loc[po, data.EFFORT], X.loc[at, data.EFFORT], X25[data.EFFORT]])
    kind = np.r_[np.zeros(len(po)), np.ones(len(at)), np.full(len(s25), 2)].astype(int)  # 0 PO, 1 AT, 2 NEON
    ids = np.r_[po, at, s25.index]
    u = zs(StandardScaler().fit_transform(E_all) @ np.array([1, 1, 1, -1]) / 4)  # access minutes is inverse
    hz = StandardScaler().fit_transform(H_all)

    # Survey design (real): Austria n visits + effort/visit; NEON real events.
    at_n = m.loc[at, "n_events"].to_numpy()
    at_le = np.log(m.loc[at, "trap_days"].to_numpy() / at_n)
    ev25 = ev25.copy()
    nidx = {sid: i for i, sid in enumerate(s25.index)}
    ev25["site"] = ev25["sample_id"].map(nidx)
    n_le = np.log(ev25["trap_hours"].to_numpy())

    partition = data.neon_cluster_folds({k: bool(v) for k, v in s25.groupby("siteID")["target"].max().items()}, 42)
    ctxs = [ctx for ctx in c.contexts if int(ctx["seed"]) == 42]

    rows = []
    t0 = time.time()
    for gamma, beta_u, design, rep in itertools.product(args.gammas, args.betas, args.designs, range(args.reps)):
        rng = np.random.default_rng(10_000 * rep + 17)
        # ---- true suitability: random 2-layer tanh net on all 105 features ----
        w_scale = 1.0 if args.truth == "smooth" else 3.0   # 3.0 saturates tanh -> strongly nonlinear eta*
        W1 = rng.normal(0, w_scale / np.sqrt(hz.shape[1]), (hz.shape[1], 32))
        w2 = rng.normal(0, 1 / np.sqrt(32), 32)
        eta_star = 1.5 * zs(zs(np.tanh(hz @ W1) @ w2) + beta_u * u)
        # ---- simulate labels ----
        logit_po = eta_star + gamma * u
        c_po = np.log(0.42 / 0.58) - np.median(logit_po)
        y_po = rng.random(len(po)) < 1 / (1 + np.exp(-(logit_po[kind == 0] + c_po)))
        z = rng.random(len(ids)) < 1 / (1 + np.exp(-(eta_star - 0.5)))
        p_at = 1 / (1 + np.exp(-(-1.4 + 0.3 * (at_le - at_le.mean()))))
        k_at = rng.binomial(at_n, p_at) * z[kind == 1]
        p_n = 1 / (1 + np.exp(-(-2.3 + 0.5 * (n_le - n_le.mean()))))
        y_ev = (rng.random(len(ev25)) < p_n) & z[kind == 2][ev25["site"].to_numpy()]
        k_n = np.bincount(ev25["site"], weights=y_ev, minlength=len(s25))

        for ctx in ctxs:
            f = int(ctx["fold"])
            tr_dev, te_dev = set(ctx["train_ids"]), set(ctx["test_ids"])
            tr = np.array([(i in tr_dev) if k < 2 else partition[s25.loc[i, "siteID"]] != f for i, k in zip(ids, kind)])
            te = np.array([(i in te_dev) if k < 2 else partition[s25.loc[i, "siteID"]] == f for i, k in zip(ids, kind)])
            use_at = design in ("AT", "AT+NEON")
            use_n = design in ("NEON", "AT+NEON")
            sv_tr = tr & (((kind == 1) & use_at) | ((kind == 2) & use_n))
            sh = StandardScaler().fit(H_all[tr & ((kind == 0) | sv_tr)])
            se = StandardScaler().fit(E_all[tr & (kind == 0)])
            po_tr = tr & (kind == 0)
            sv_idx = np.flatnonzero(sv_tr)
            # events for training survey sites
            loc = {g: j for j, g in enumerate(sv_idx)}
            at_pos = np.flatnonzero(kind == 1)
            n_pos = np.flatnonzero(kind == 2)
            ev_site, ev_le, ev_p, ev_n = [], [], [], []
            for j, g in enumerate(sv_idx):
                if kind[g] == 1:
                    a = np.searchsorted(at_pos, g)
                    ev_site.append(j); ev_le.append(at_le[a] - at_le.mean())
                    ev_p.append(k_at[a]); ev_n.append(at_n[a] - k_at[a])
            if use_n:
                ns = ev25["site"].to_numpy()
                keep = np.isin(n_pos[ns], sv_idx)
                for e_i in np.flatnonzero(keep):
                    ev_site.append(loc[n_pos[ns[e_i]]]); ev_le.append(n_le[e_i] - n_le.mean())
                    ev_p.append(float(y_ev[e_i])); ev_n.append(1.0 - float(y_ev[e_i]))
            det = np.array([(k_at[np.searchsorted(at_pos, g)] if kind[g] == 1 else k_n[np.searchsorted(n_pos, g)]) > 0
                            for g in sv_idx])
            src = (kind[sv_idx] == 2).astype(np.int64)
            batch = dict(
                po_h=sh.transform(H_all[po_tr]).astype(np.float32),
                po_e=se.transform(E_all[po_tr]).astype(np.float32),
                po_y=y_po[np.flatnonzero(po_tr)].astype(np.float32),  # PO sites occupy the first len(po) rows
                sv_h=sh.transform(H_all[sv_idx]).astype(np.float32), sv_src=src, sv_det=det,
                ev_site=np.asarray(ev_site, np.int64), ev_logeff=np.asarray(ev_le, np.float32),
                ev_pos=np.asarray(ev_p, np.float32), ev_neg=np.asarray(ev_n, np.float32), n_survey=2,
            )
            Hte = sh.transform(H_all[te]).astype(np.float32)
            eff_idx = [cols.index(e) for e in data.EFFORT]
            Ez = lambda M: se.transform(M)  # noqa: E731
            po_o, sv_o, te_o = orth_rep(sh.transform(H_all[tr & ((kind == 0) | sv_tr)]), Ez(E_all[tr & ((kind == 0) | sv_tr)]),
                                        [sh.transform(H_all[po_tr]), sh.transform(H_all[sv_idx]), sh.transform(H_all[te])],
                                        [Ez(E_all[po_tr]), Ez(E_all[sv_idx]), Ez(E_all[te])], eff_idx)
            batch_o = {**batch, "po_h": po_o, "sv_h": sv_o}
            te_sv = te & (kind > 0)
            for name in args.models:
                cfg = replace(MODELS[name], l2=args.l2)
                bt, ht = (batch_o, te_o) if name in ORTH else (batch, Hte)
                eh = np.mean([predict_eta(fit(cfg, bt, seed=100 * rep + 10 * f + k), ht) for k in range(args.members)], axis=0)
                es = eta_star[te]
                urb = u[te] > 0
                rows.append({
                    "truth": args.truth, "gamma": gamma, "beta_u": beta_u, "design": design, "rep": rep, "context": ctx["id"], "model": name,
                    "rho_all": spearmanr(eh, es).statistic,
                    "rho_urban": spearmanr(eh[urb], es[urb]).statistic,
                    "leak": np.corrcoef(zs(eh) - zs(es), u[te])[0, 1],
                    "AUC_z": roc_auc_score(z[te_sv], eh[(kind[te] > 0)]),
                })
        print(f"gamma={gamma} beta_u={beta_u} design={design} rep={rep} ({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(args.out / "results.csv", index=False)
    summ = res.groupby(["gamma", "beta_u", "design", "model"])[["rho_all", "rho_urban", "leak", "AUC_z"]].mean().round(3)
    summ.to_csv(args.out / "summary.csv")
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        print(summ)


if __name__ == "__main__":
    main()
