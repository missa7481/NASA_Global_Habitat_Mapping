#!/usr/bin/env python3
"""RSI iteration 6: is HiCo's hierarchy worth anything once a strong niche model exists?

Question (user, after I1-I5): RF beats HiCo on every holdout design. HiCo was built for global-to-local
coherence, which accuracy alone does not test. Does coherence cost accuracy, and does HiCo's hierarchy add
anything beyond a post-hoc reconciliation of a strong flat model?

Coherence (HiCo's definition): for every 0.5-degree parent P and class s,
    mean_{x in P} exp(eta_s(x)) == exp(A_s(P))
i.e. the coarse (parent) map is exactly the average of the fine map over its children.

Arms (same splits, scored at the fine cell level AND the coarse parent level):
  RF           flat RF at fine level; the coarse map is an independently fitted parent-level RF
               (what a practitioner gets by mapping each scale separately) -> can contradict.
  RF-agg       coarse map = average of RF fine map over children (trivially coherent; reported at coarse level only).
  RF+Recon     post-hoc HiCo normalisation: eta = A_RFcoarse(P) + B_RF(x) - logmeanexp_P(B_RF)
  HiCo         original HiCo-Fuse (macro MLP on parent-mean features + local MLP), 2 seeds
  HiCo-RF      HiCo likelihood with the local term fixed to RF and the macro network A learned jointly
               through the fine-level likelihood, 2 seeds
Prediction: if HiCo's value is coherence, RF+Recon or HiCo-RF should keep RF accuracy (fine) while being coherent;
HiCo's own deficit would then be its neural local term, not the hierarchy.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.common import ART, DEV, Data, designs, fit_net, fit_sklearn, inner_split, metrics, predict_sklearn  # noqa: E402
from hicofuse.models import mlp, multinomial_nll, parent_logsumexp  # noqa: E402
from hicofuse.train import Problem  # noqa: E402

OUT = ART / "it6"
OUT.mkdir(parents=True, exist_ok=True)
DESIGNS = ("block10", "loro", "env", "zone")
SEEDS = (0, 1)


def eta_of(P):
    P = np.clip(P, 1e-4, 1)
    return np.log(P[:, :4]) - np.log(P[:, 4:5])


def softmax_eta(eta):
    z = np.c_[eta, np.zeros(len(eta))]
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def logmeanexp_by(B, g, n):
    t = torch.as_tensor(B, device=DEV, dtype=torch.float32)
    lse = parent_logsumexp(t, torch.as_tensor(g, device=DEV), n).cpu().numpy()
    return lse - np.log(np.bincount(g, minlength=n))[:, None]


def pool_parent(Y, gpar, n):
    out = np.zeros((n, Y.shape[1]), np.float32)
    np.add.at(out, gpar, Y)
    return out


def fit_macro_given_B(st, par_ids, pall, Bn_obs, par_obs, Y, itr, iva, seed, epochs=1500, patience=60):
    """Learn HiCo's macro net A with the (normalised) local term fixed: eta = A(P) + Bn(x)."""
    torch.manual_seed(seed)
    A = mlp(pall.shape[1], 4, hidden=32, p=0.1).to(DEV)
    opt = torch.optim.AdamW(A.parameters(), lr=3e-3, weight_decay=1e-4)
    Xp = torch.as_tensor(pall, device=DEV)
    Bn = torch.as_tensor(Bn_obs, device=DEV, dtype=torch.float32)
    po = torch.as_tensor(par_obs, device=DEV)
    Yt = torch.as_tensor(Y, device=DEV)
    it, iv = torch.as_tensor(np.flatnonzero(itr), device=DEV), torch.as_tensor(np.flatnonzero(iva), device=DEV)
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        A.train()
        opt.zero_grad()
        e = A(Xp)[po] + Bn
        multinomial_nll(e[it], Yt[it]).backward()
        opt.step()
        if ep % 5 == 4:
            A.eval()
            with torch.no_grad():
                v = float(multinomial_nll((A(Xp)[po] + Bn)[iv], Yt[iv]))
            if v < best - 1e-4:
                best, bad, state = v, 0, {k: t.clone() for k, t in A.state_dict().items()}
            else:
                bad += 5
                if bad >= patience:
                    break
    A.load_state_dict(state)
    A.eval()
    with torch.no_grad():
        return A(Xp).cpu().numpy()


def contradiction(Pc, Pf_agg):
    tv = 0.5 * np.abs(Pc - Pf_agg).sum(1)
    dom = (Pc.argmax(1) != Pf_agg.argmax(1))
    return float(tv.mean()), float(dom.mean())


def main():
    t0 = time.time()
    D = Data()
    st = D.st
    DS = designs(D)
    par_obs_all = st.par[D.pos]                                   # global parent id of each report cell
    rows, coh = [], []
    for dname in DESIGNS:
        for si, (tr, Ytr, te, Yte, label) in enumerate(DS[dname]):
            sid = 600 + si
            # ---- parent universe for this split: parents of any report cell (train or test)
            par_ids = np.unique(par_obs_all)
            gp = np.searchsorted(par_ids, par_obs_all)            # local parent index per report cell
            child = np.flatnonzero(np.isin(st.par, par_ids))
            gchild = np.searchsorted(par_ids, st.par[child])
            n = len(par_ids)
            pall = st.par_all[par_ids]
            logn = np.log(st.par_n[par_ids])
            # ---- RF fine
            rf = fit_sklearn("RF", D.X[tr], Ytr, seed=si)
            P_rf_te = predict_sklearn(rf, D.X[te])
            Bc = np.concatenate([eta_of(predict_sklearn(rf, st.X[child[i:i + 200_000]])) for i in range(0, len(child), 200_000)])
            lme = logmeanexp_by(Bc, gchild, n)                    # log mean_children exp(B) per parent
            # report cells as children: position of each report cell inside `child`
            loc = np.searchsorted(child, D.pos)
            B_obs = Bc[loc]
            Bn_obs = B_obs - lme[gp]
            # ---- RF coarse (independent parent-level model on pooled training counts)
            Ytr_par = pool_parent(Ytr, gp[tr], n)
            ptr = np.flatnonzero(Ytr_par.sum(1) > 0)
            rfc = fit_sklearn("RF", pall[ptr], Ytr_par[ptr], seed=si)
            A_rfc = eta_of(predict_sklearn(rfc, pall))
            # ---- test parents (pooled test counts)
            Yte_par = pool_parent(Yte, gp[te], n)
            pte = np.flatnonzero(Yte_par.sum(1) > 0)
            Yc = Yte_par[pte]
            res_f, res_c = {}, {}
            res_f["RF"] = P_rf_te
            res_c["RF"] = softmax_eta(A_rfc[pte])
            res_c["RF-agg"] = softmax_eta(lme[pte])
            res_f["RF+Recon"] = softmax_eta(A_rfc[gp[te]] + Bn_obs[te])
            res_c["RF+Recon"] = res_c["RF"]
            tv, dom = contradiction(res_c["RF"], res_c["RF-agg"])
            coh.append({"design": dname, "split": label, "model": "RF (separate scales)", "TV": tv, "dominant_flip": dom})
            # ---- HiCo-RF: macro A learned through the fine-level likelihood, local term fixed to RF
            itr, iva = inner_split(D, tr, seed=sid)
            Pf, Pc = [], []
            for s in SEEDS:
                A = fit_macro_given_B(st, par_ids, pall, Bn_obs[tr], gp[tr], Ytr, itr, iva, seed=sid + s)
                Pf.append(softmax_eta(A[gp[te]] + Bn_obs[te]))
                Pc.append(softmax_eta(A[pte]))
            res_f["HiCo-RF"], res_c["HiCo-RF"] = np.mean(Pf, 0), np.mean(Pc, 0)
            # ---- HiCo original
            Pf, Pc, chk = [], [], []
            for s in SEEDS:
                net = fit_net("HiCo", D, tr[itr], Ytr[itr], tr[iva], Ytr[iva], seed=sid + s)
                with torch.no_grad():
                    pb = Problem(st, D.pos[te], Yte)
                    e, A = net.eta_hier(pb.child_b, pb.child_par, pb.n_par, pb.pall, pb.logn)
                    e_te = e[pb.obs_in_child].cpu().numpy()
                    A = A.cpu().numpy()
                    # coherence check: mean_children exp(eta) vs exp(A)
                    lme_h = logmeanexp_by(e.cpu().numpy(), pb.child_par.cpu().numpy(), pb.n_par)
                    chk.append(float(np.abs(lme_h - A).max()))
                tpar = np.unique(st.par[D.pos[te]])               # Problem's parent order
                mp = np.searchsorted(tpar, par_ids[pte])
                Pf.append(softmax_eta(e_te))
                Pc.append(softmax_eta(A[mp]))
                del net, pb
                torch.cuda.empty_cache()
            res_f["HiCo"], res_c["HiCo"] = np.mean(Pf, 0), np.mean(Pc, 0)
            coh.append({"design": dname, "split": label, "model": "HiCo (max |log-mean-exp - A|)", "TV": max(chk), "dominant_flip": 0.0})
            for m, P in res_f.items():
                rows.append({"design": dname, "split": label, "model": m, "level": "fine", **metrics(P, Yte)})
            for m, P in res_c.items():
                rows.append({"design": dname, "split": label, "model": m, "level": "coarse", **metrics(P, Yc)})
            print(f"{dname:8s} {label:14s} n_te={len(te)} parents_te={len(pte)} children={len(child)}  {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
            pd.DataFrame(coh).to_csv(OUT / "coherence.csv", index=False)
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 250):
        print(df.groupby(["level", "model", "design"])[["AUC_mean", "nll"]].mean().unstack("design").round(3))
        print(pd.DataFrame(coh).groupby(["model", "design"])[["TV", "dominant_flip"]].mean().round(3))
    tests = []
    for lev in ("fine", "coarse"):
        w = df[df.level == lev].pivot_table(index=["design", "split"], columns="model", values=["AUC_mean", "nll"])
        for m in w["nll"].columns:
            if m == "RF":
                continue
            for met, better in (("AUC_mean", 1), ("nll", -1)):
                d = (w[met][m] - w[met]["RF"]).dropna() * better
                k = int((d > 0).sum())
                tests.append({"level": lev, "model": m, "metric": met, "wins_vs_RF": k, "n": len(d),
                              "mean_diff": float(d.mean() * better), "p_sign": binomtest(k, len(d)).pvalue})
    T = pd.DataFrame(tests)
    T.to_csv(OUT / "paired_vs_rf.csv", index=False)
    print(T.round(3).to_string())


if __name__ == "__main__":
    main()
