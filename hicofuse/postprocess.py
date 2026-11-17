#!/usr/bin/env python3
"""External validation, cross-level coherence audit and disease layers for HiCo-Fuse.

Outputs (artifacts/hicofuse/results/):
  external_validation.csv   Kraemer compendium (all / >~110 km from iNat reports) and NEON 2026
  coherence_audit.csv       separately-fitted coarse model vs aggregated fine map, vs HiCo
  layers.npz                per-cell suitability percentiles, thermal indices, disease indices
Thermal envelopes (Tmin, Topt, Tmax, deg C):
  dengue / Ae. aegypti (17.8, 29.1, 34.6) and Ae. albopictus (16.2, 26.4, 31.6)  Mordecai et al. 2017
  malaria / Anopheles (16.0, 25.0, 34.0)                                        Mordecai et al. 2013
  West Nile / Culex (16.0, 24.5, 33.0)  APPROXIMATE from Shocket et al. 2020; verify before use.
Thermal index = mean of envelope values at BIO10 (warmest quarter), BIO1 (annual), BIO11 (coldest
quarter): a coarse proxy for the share of the year that is thermally permissive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicofuse.models import Net, multinomial_nll  # noqa: E402
from hicofuse.train import ART, DEV, Store  # noqa: E402

G = ART / "global"
R = ART / "results"
MODELS = ["MLPEarly", "GatedMoE", "HiCo", "HiCoMoE", "HF2"]   # HF2 = HiCo-Fuse 2 (rsi/hf2_global.py)
CL = ["Ae_albopictus", "Ae_aegypti", "Culex", "Anopheles"]
ENV = {"aegypti": (17.8, 29.1, 34.6), "albopictus": (16.2, 26.4, 31.6), "malaria": (16.0, 25.0, 34.0), "wnv": (16.0, 24.5, 33.0)}


def envelope(T, tmin, topt, tmax):
    T = np.asarray(T, float)
    up = 1 - ((T - topt) / (topt - tmin)) ** 2
    dn = 1 - ((T - topt) / (tmax - topt)) ** 2
    out = np.where(T <= topt, up, dn)
    return np.clip(np.where((T > tmin) & (T < tmax), out, 0), 0, 1)


def pct(v):
    r = np.empty(len(v))
    r[np.argsort(v, kind="stable")] = np.arange(len(v))
    return (r + 0.5) / len(v)


def main():
    R.mkdir(exist_ok=True)
    st = Store()
    d = st.d
    rc = st.rc
    feats = st.info["features"]
    eta = {m: np.load(G / f"{m}_eta.npy") for m in MODELS if (G / f"{m}_eta.npy").exists()}
    sup = np.load(G / "support.npy")

    def share(e):
        z = np.c_[e, np.zeros(len(e))].astype(np.float64)
        z = np.exp(z - z.max(1, keepdims=True))
        return (z / z.sum(1, keepdims=True))[:, :4]

    two_part = {m: (sup[:, None] * share(e)).astype(np.float32) for m, e in eta.items()}
    for m, v in two_part.items():
        np.save(G / f"{m}_twopart.npy", v)
    scores = {**{f"{m} (composition only)": e for m, e in eta.items()},
              **{f"{m} x support (two-part)": v for m, v in two_part.items()}}

    # ---------------- external validation ----------------
    rng = np.random.default_rng(5)
    bg = rng.choice(len(st.cell), 50_000, replace=False)
    obs_pos = st.pos[d["obs_idx"]]
    rows = []
    pr, pc = rc[:, 0] // 10, rc[:, 1] // 10
    for s, sp in ((0, "albopictus"), (1, "aegypti")):
        kpos = st.pos[d[f"kraemer_{sp}"]]
        kpos = kpos[kpos >= 0]
        inat = obs_pos[d["Y"][:, s] > 0]
        near = set()
        for a, b in zip(pr[inat], pc[inat]):
            for da in (-2, -1, 0, 1, 2):
                for db in (-2, -1, 0, 1, 2):
                    near.add((a + da, b + db))
        far = np.array([(a, b) not in near for a, b in zip(pr[kpos], pc[kpos])])
        for m, e in scores.items():
            for subset, kp in (("all", kpos), (">~110 km from iNat reports", kpos[far])):
                y = np.r_[np.ones(len(kp)), np.zeros(len(bg))]
                rows.append({"dataset": f"Kraemer Ae. {sp}", "subset": subset, "model": m, "n_pos": len(kp),
                             "AUC": roc_auc_score(y, np.r_[e[kp, s], e[bg, s]])})
    # NEON 2026 (Ae. albopictus detection / non-detection)
    ne = pd.read_csv(Path(__file__).resolve().parents[2] / "webapp/runtime/candidates/admission_2026_11_12/global_external2026/external_evaluated_sites.csv")
    ne = ne[ne["independent_gt10km"].astype(bool)]
    r_ = np.floor((85 - ne.latitude) / 0.05).astype(int)
    c_ = np.floor((ne.longitude + 180) / 0.05).astype(int)
    key = rc[:, 0].astype(np.int64) * 7200 + rc[:, 1]
    order = np.argsort(key)
    k = r_.to_numpy() * 7200 + c_.to_numpy()
    p = np.searchsorted(key[order], k)
    ok = key[order][np.clip(p, 0, len(key) - 1)] == k
    npos = order[np.clip(p, 0, len(key) - 1)][ok]
    yN = ne["target"].to_numpy()[ok]
    for m, e in scores.items():
        rows.append({"dataset": "NEON 2026 Ae. albopictus", "subset": f"{ok.sum()} sites", "model": m,
                     "n_pos": int(yN.sum()), "AUC": roc_auc_score(yN, e[npos, 0])})
    ext = pd.DataFrame(rows)
    ext.to_csv(R / "external_validation.csv", index=False)
    print(ext.pivot_table(index=["dataset", "subset"], columns="model", values="AUC").round(3))

    # ---------------- coherence audit ----------------
    # Separately fitted coarse model: parent-level composition from parent-mean features.
    par_obs = st.par[obs_pos]
    up, inv = np.unique(par_obs, return_inverse=True)
    Yp = np.zeros((len(up), 5), np.float32)
    np.add.at(Yp, inv, d["Y"])
    torch.manual_seed(0)
    coarse = Net("MLPEarly", {m: len(c) for m, c in st.cols.items()}).to(DEV)
    opt = torch.optim.AdamW(coarse.parameters(), lr=3e-3, weight_decay=1e-4)
    Xp = torch.as_tensor(st.par_all[up], device=DEV)
    Yt = torch.as_tensor(Yp, device=DEV)
    for _ in range(300):
        opt.zero_grad()
        multinomial_nll(coarse.flat({"all": Xp}), Yt).backward()
        opt.step()
    coarse.eval()
    with torch.no_grad():
        eta_coarse = coarse.flat({"all": torch.as_tensor(st.par_all, device=DEV)}).cpu().numpy()
    n_par = len(st.par_keys)
    coh = []
    for m, e in eta.items():
        agg = np.zeros((n_par, 4))
        for s in range(4):
            agg[:, s] = np.bincount(st.par, weights=np.exp(e[:, s].astype(np.float64)), minlength=n_par) / st.par_n
        if m.startswith("HiCo") or m == "HF2":
            ref, label = np.load(G / f"{m}_parentA.npy"), "own macro term A (intensity-space ensemble)"
        else:
            ref, label = eta_coarse, "separately fitted 0.5-deg model"
        for s, name in enumerate(CL):
            lr = np.log(agg[:, s]) - ref[:, s]
            coh.append({"model": m, "coarse_reference": label, "class": name,
                        "median_abs_log_ratio": float(np.median(np.abs(lr))),
                        "share_parents_child_mean_exceeds_coarse_by_10pct": float(np.mean(lr > np.log(1.1))),
                        "share_parents_off_by_more_than_2x": float(np.mean(np.abs(lr) > np.log(2))),
                        "spearman_coarse_vs_aggregated": float(spearmanr(ref[:, s], np.log(agg[:, s])).statistic)})
    # exact single-seed HiCo check (deterministic identity)
    coh_df = pd.DataFrame(coh)
    coh_df.to_csv(R / "coherence_audit.csv", index=False)
    print(coh_df.round(4).to_string())

    # ---------------- layers ----------------
    f = {n: st.X[:, i] * st.sd[i] + st.mu[i] for i, n in enumerate(feats) if n in ("bio19_1", "bio19_10", "bio19_11")}
    temps = [f["bio19_10"], f["bio19_1"], f["bio19_11"]]
    therm = {k: np.mean([envelope(t, *v) for t in temps], axis=0).astype(np.float32) for k, v in ENV.items()}
    base_model = "HF2" if "HF2" in two_part else "HiCo"
    base = two_part[base_model]
    suit = {name: pct(base[:, s]).astype(np.float32) for s, name in enumerate(CL)}
    dengue = 1 - (1 - suit["Ae_aegypti"] * therm["aegypti"]) * (1 - suit["Ae_albopictus"] * therm["albopictus"])
    np.savez_compressed(R / "layers.npz", **{f"suit_{k}": v for k, v in suit.items()},
                        **{f"therm_{k}": v for k, v in therm.items()},
                        dengue=dengue.astype(np.float32),
                        wnv=(suit["Culex"] * therm["wnv"]).astype(np.float32),
                        malaria=(suit["Anopheles"] * therm["malaria"]).astype(np.float32))
    (R / "layers_meta.json").write_text(json.dumps({"base_model": f"{base_model} two-part (support x share)", "suitability": "percentile of S_s among land cells",
                                                    "envelopes": ENV, "wnv_parameters": "approximate; verify"}, indent=1))
    print("layers saved")


if __name__ == "__main__":
    main()
