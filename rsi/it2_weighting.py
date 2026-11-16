#!/usr/bin/env python3
"""RSI iteration 2: should information sources be weighted differently, and should weights vary in space?

Experts (each a 2x64 MLP on one modality, 3 seeds) + a spatial expert (kNN-geo).
Combination = log-linear pool  log p(x) = sum_m w_m(x) log p_m(x) + b, renormalised.
Strategies
  Equal          w_m = 1/M
  Theory         fixed a priori: Climate .35, Landscape .25, Human .20, AlphaEarth .20, Geo 0
                 (niche theory: climate bounds the range; land cover / settlement modulate; interpolation untrusted)
  Stack-global   w >= 0 learned on inner out-of-fold expert predictions (one weight vector per class)
  Stack-region   global weights + region offsets shrunk to 0 (unseen region -> global weights)
  Attn-env       w(x) = softmax(gate(climate, human context)), learned on inner OOF
  Attn-support   gate also sees data-support context: log distance to training, log density, env novelty
  Uncertainty    w_m(x) proportional to 1 / (seed / bootstrap variance of expert m at x)
  Reliability    w_m(x) proportional to exp(-tau * local inner-OOF NLL of expert m among the 50 env-nearest cells)
Every learned weighting is fitted on inner OOF predictions of the TRAINING set only, so the generalisation
gap (inner gain vs outer gain over Equal) measures overfitting of the weighting itself.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.common import (ART, DEV, MODS, REGIONS, Data, cell_nll, designs, fit_flat, inner_split, knn_geo,  # noqa: E402
                        metrics, predict_flat)
from rsi.support_ctx import env_space, support_context  # noqa: E402

OUT = ART / "it2"
EXPERTS = MODS + ["Geo"]
THEORY = np.array([0.35, 0.25, 0.20, 0.20, 0.0])
SEEDS = (0, 1, 2)


def expert_preds(D, tr, Ytr, q, sid, n_seeds=3):
    """Log-probs (n_q, M, 5) and variance proxy (n_q, M) for every expert, trained on tr."""
    itr, iva = inner_split(D, tr, seed=sid)
    L, V = [], []
    for m in MODS:
        X = D.X[:, D.cols[m]]
        ps = np.stack([np.log(np.clip(predict_flat(fit_flat(X[tr][itr], Ytr[itr], X[tr][iva], Ytr[iva], seed=sid + s), X[q]), 1e-9, 1))
                       for s in range(n_seeds)])
        L.append(ps.mean(0))
        V.append(ps.var(0).mean(1) if n_seeds > 1 else np.zeros(len(q)))
    P, _ = knn_geo(D, tr, Ytr, q)
    L.append(np.log(np.clip(P, 1e-9, 1)))
    if n_seeds > 1:
        rng = np.random.default_rng(sid)
        bs = [np.log(np.clip(knn_geo(D, tr[b], Ytr[b], q)[0], 1e-9, 1)) for b in
              (rng.choice(len(tr), len(tr), replace=True) for _ in range(8))]
        V.append(np.stack(bs).var(0).mean(1))
    else:
        V.append(np.zeros(len(q)))
    return np.stack(L, 1).astype(np.float32), np.stack(V, 1).astype(np.float32)


def inner_oof(D, tr, Ytr, sid, k=4):
    b = (D.rc[tr, 0] // 200).astype(np.int64) * 100 + D.rc[tr, 1] // 200
    ub = np.unique(b)
    fb = dict(zip(ub, np.random.default_rng(sid).permutation(len(ub)) % k))
    f = np.array([fb[x] for x in b])
    L = np.zeros((len(tr), len(EXPERTS), 5), np.float32)
    for j in range(k):
        a, h = f != j, f == j
        L[h], _ = expert_preds(D, tr[a], Ytr[a], tr[h], sid + 7 * j, n_seeds=1)
    return L, f


def pool(L, w, b=None):
    """L (n, M, 5); w (n, M, 5) or broadcastable; returns probs."""
    z = (L * w).sum(1)
    if b is not None:
        z = z + b
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


class Gate(nn.Module):
    def __init__(self, d_ctx, M):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(d_ctx, 32), nn.SiLU(), nn.Linear(32, M))
        self.scale = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(5))

    def forward(self, L, ctx):
        w = torch.softmax(self.g(ctx), -1) * F.softplus(self.scale) * 1.4
        z = (L * w[..., None]).sum(1) + self.b
        return z, w


def nll_t(z, Y):
    return -(Y * F.log_softmax(z, 1)).sum() / Y.sum()


def fit_stack(L, Y, groups=None, n_groups=0, lam=1.0, epochs=600):
    """Non-negative per-class weights (+ optional shrunk group offsets) on log-probs."""
    Lt, Yt = torch.as_tensor(L, device=DEV), torch.as_tensor(Y, device=DEV)
    M = L.shape[1]
    raw = torch.full((M, 5), -1.0, device=DEV, requires_grad=True)
    b = torch.zeros(5, device=DEV, requires_grad=True)
    params = [raw, b]
    if groups is not None:
        dg = torch.zeros((n_groups, M, 5), device=DEV, requires_grad=True)
        gt = torch.as_tensor(groups, device=DEV)
        params.append(dg)
    opt = torch.optim.Adam(params, lr=0.05)
    for _ in range(epochs):
        opt.zero_grad()
        w = F.softplus(raw)[None]
        if groups is not None:
            w = F.softplus(raw[None] + dg[gt])
        loss = nll_t((Lt * w).sum(1) + b, Yt)
        if groups is not None:
            loss = loss + lam * 1e-2 * dg.pow(2).mean()
        loss.backward()
        opt.step()
    out = {"w": F.softplus(raw).detach().cpu().numpy(), "b": b.detach().cpu().numpy()}
    if groups is not None:
        out["dg"] = dg.detach().cpu().numpy()
        out["raw"] = raw.detach().cpu().numpy()
    return out


def fit_gate(L, Y, ctx, f_inner, seed=0, epochs=1500):
    torch.manual_seed(seed)
    Lt, Yt, Ct = (torch.as_tensor(a, device=DEV) for a in (L, Y, ctx))
    va = torch.as_tensor(f_inner == 0, device=DEV)
    g = Gate(ctx.shape[1], L.shape[1]).to(DEV)
    opt = torch.optim.AdamW(g.parameters(), lr=3e-3, weight_decay=1e-3)
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        opt.zero_grad()
        z, _ = g(Lt[~va], Ct[~va])
        nll_t(z, Yt[~va]).backward()
        opt.step()
        if ep % 5 == 4:
            with torch.no_grad():
                v = float(nll_t(g(Lt[va], Ct[va])[0], Yt[va]))
            if v < best - 1e-5:
                best, bad, state = v, 0, {k: t.detach().clone() for k, t in g.state_dict().items()}
            else:
                bad += 5
                if bad > 100:
                    break
    g.load_state_dict(state)
    return g.eval()


def gate_apply(g, L, ctx):
    with torch.no_grad():
        z, w = g(torch.as_tensor(L, device=DEV), torch.as_tensor(ctx, device=DEV))
        z = z.cpu().numpy()
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True), w.cpu().numpy()


def reliability_weights(Lq, Eq, Ein, Lin, Yin, tau, k=50, exclude_self=False):
    nn_ = NearestNeighbors(n_neighbors=k + int(exclude_self)).fit(Ein)
    _, ind = nn_.kneighbors(Eq)
    if exclude_self:
        ind = ind[:, 1:]
    Pin = np.exp(Lin)
    nl = np.stack([cell_nll(Pin[:, m], Yin) for m in range(Lin.shape[1])], 1)   # n_in, M
    R = nl[ind].mean(1)                                                        # n_q, M
    w = np.exp(-tau * (R - R.min(1, keepdims=True)))
    return w / w.sum(1, keepdims=True)


def main(dlist=("block10", "loro", "env")):
    OUT.mkdir(parents=True, exist_ok=True)
    D = Data()
    DS = designs(D)
    E = env_space(D)
    ctx_env = np.c_[D.X[:, D.cols["Climate"]], D.X[:, D.cols["Human"]]].astype(np.float32)
    reg_id = np.array([REGIONS.index(r) for r in D.region])
    rows, wrows, gate_rows = [], [], []
    t0 = time.time()
    for dname in dlist:
        for si, (tr, Ytr, te, Yte, label) in enumerate(DS[dname]):
            sid = 5000 + 31 * si + 7 * len(dname)
            Lin, finner = inner_oof(D, tr, Ytr, sid)
            Lte, Vte = expert_preds(D, tr, Ytr, te, sid)
            # support context: inner cells vs their own inner-train fold; test cells vs full training set
            Sin = np.zeros((len(tr), 3), np.float32)
            for j in np.unique(finner):
                h = finner == j
                c = support_context(D, tr[~h], tr[h], E)
                Sin[h] = np.c_[np.log1p(c["dist_km"]), np.log1p(c["dens250"]), np.log(c["env_nov"] + 1e-3)]
            c = support_context(D, tr, te, E)
            Ste = np.c_[np.log1p(c["dist_km"]), np.log1p(c["dens250"]), np.log(c["env_nov"] + 1e-3)].astype(np.float32)
            mu, sd = Sin.mean(0), Sin.std(0) + 1e-6
            Sin_s, Ste_s = (Sin - mu) / sd, (Ste - mu) / sd
            preds, inner = {}, {}
            M = len(EXPERTS)
            eq = np.full((1, M, 1), 1.0 / M)
            preds["Equal"], inner["Equal"] = pool(Lte, eq), pool(Lin, eq)
            th = THEORY[None, :, None]
            preds["Theory"], inner["Theory"] = pool(Lte, th), pool(Lin, th)
            sg = fit_stack(Lin, Ytr)
            preds["Stack-global"], inner["Stack-global"] = pool(Lte, sg["w"][None], sg["b"]), pool(Lin, sg["w"][None], sg["b"])
            sr = fit_stack(Lin, Ytr, groups=reg_id[tr], n_groups=len(REGIONS))
            wr_tr = np.log1p(np.exp(sr["raw"][None] + sr["dg"][reg_id[tr]]))
            seen = np.isin(reg_id[te], np.unique(reg_id[tr]))
            wr_te = np.where(seen[:, None, None], np.log1p(np.exp(sr["raw"][None] + sr["dg"][reg_id[te]])), np.log1p(np.exp(sr["raw"]))[None])
            preds["Stack-region"], inner["Stack-region"] = pool(Lte, wr_te, sr["b"]), pool(Lin, wr_tr, sr["b"])
            for nm, Cin, Cte in (("Attn-env", ctx_env[tr], ctx_env[te]),
                                 ("Attn-support", np.c_[ctx_env[tr], Sin_s], np.c_[ctx_env[te], Ste_s])):
                g = fit_gate(Lin, Ytr, Cin.astype(np.float32), finner, seed=sid)
                preds[nm], wte = gate_apply(g, Lte, Cte.astype(np.float32))
                inner[nm], _ = gate_apply(g, Lin, Cin.astype(np.float32))
                if nm == "Attn-support":
                    for m in range(M):
                        gate_rows.append(pd.DataFrame({"design": dname, "split": label, "expert": EXPERTS[m], "w": wte[:, m],
                                                       "log_dist": Ste[:, 0], "log_dens": Ste[:, 1], "log_envnov": Ste[:, 2]}))
            uw = 1.0 / (Vte + np.median(Vte, 0, keepdims=True) + 1e-6)
            uw = uw / uw.sum(1, keepdims=True)
            preds["Uncertainty"] = pool(Lte, uw[..., None])
            # reliability: tau tuned on inner OOF (leave-self-out neighbourhoods)
            best_tau, best_v = 0.0, np.inf
            for tau in (0.0, 0.5, 1, 2, 4, 8):
                wi = reliability_weights(Lin, E[tr], E[tr], Lin, Ytr, tau, exclude_self=True)
                v = metrics(pool(Lin, wi[..., None]), Ytr)["nll"]
                if v < best_v:
                    best_tau, best_v = tau, v
            wi = reliability_weights(Lin, E[tr], E[tr], Lin, Ytr, best_tau, exclude_self=True)
            inner["Reliability"] = pool(Lin, wi[..., None])
            preds["Reliability"] = pool(Lte, reliability_weights(Lte, E[te], E[tr], Lin, Ytr, best_tau)[..., None])
            for m in range(M):
                e1 = np.zeros((1, M, 1))
                e1[0, m] = 1
                preds[f"Only:{EXPERTS[m]}"], inner[f"Only:{EXPERTS[m]}"] = pool(Lte, e1), pool(Lin, e1)
            for nm, P in preds.items():
                r = {"design": dname, "split": label, "strategy": nm, **metrics(P, Yte)}
                if nm in inner:
                    r["inner_nll"] = metrics(inner[nm], Ytr)["nll"]
                rows.append(r)
            wrows.append({"design": dname, "split": label, "tau": best_tau,
                          **{f"w_{EXPERTS[m]}": float(sg["w"][m, :4].mean()) for m in range(M)}})
            print(f"{dname} {label} {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
            pd.DataFrame(wrows).to_csv(OUT / "stack_weights.csv", index=False)
    pd.concat(gate_rows).to_csv(OUT / "gate_weights.csv.gz", index=False)
    res = pd.DataFrame(rows)
    with pd.option_context("display.width", 250):
        print(res.groupby(["strategy", "design"])[["AUC_mean", "nll"]].mean().unstack("design").round(3))


if __name__ == "__main__":
    main()
