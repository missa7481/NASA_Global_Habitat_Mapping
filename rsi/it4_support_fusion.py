#!/usr/bin/env python3
"""RSI iteration 4: support-matched fusion.

Evidence motivating it (I2-01, I2-02):
  * every learned weighting gained on inner OOF (+0.04..+0.11 NLL) and lost under leave-region-out (-0.02..-0.20);
  * the inner OOF was produced by 10-degree blocks inside the training regions, i.e. an interpolation regime,
    so stacking learned to trust the spatial interpolator (w_Geo ~0.45-0.49);
  * the support-aware gate did NOT down-weight Geo with distance: it had never seen far-from-data inner cells.

Two hypotheses, tested factorially:
  H4a (validation design): weights learned on inner OOF that spans the deployment regime (inner 10-degree
       blocks + inner leave-one-training-region-out) transfer better than weights learned on block OOF alone.
  H4b (structure): a monotone support-decay combiner transfers better than a free gate:
         log p(x) ~ sum_m w_m k_m(x) log p_m(x) + (1 - sum_m w_m k_m(x))_+ log prior + b
         k_Geo(x) = exp(-d(x)/rho)                (trust interpolation only within a correlation range rho)
         k_m(x)   = exp(-beta_m * novelty(x))      (covariate experts fade towards the prior when novel)
       ~15 parameters; monotonicity is the ecological/spatial assumption.
Experts: Climate, Landscape, Human, AlphaEarth, All-feature MLP (early fusion), Geo (kNN).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.common import ART, DEV, MODS, REGIONS, Data, designs, fit_flat, inner_split, knn_geo, metrics, predict_flat  # noqa: E402
from rsi.it2_weighting import fit_gate, fit_stack, gate_apply, pool  # noqa: E402
from rsi.support_ctx import env_space, support_context  # noqa: E402

OUT = ART / "it4"
EXPERTS = MODS + ["All", "Geo"]


def expert_logp(D, tr, Ytr, q, sid, n_seeds=1):
    itr, iva = inner_split(D, tr, seed=sid)
    if iva.sum() < 20 or itr.sum() < 50:
        iva = np.random.default_rng(sid).random(len(tr)) < 0.15
        itr = ~iva
    L = []
    allc = sum((D.cols[m] for m in MODS), [])
    for cols in [D.cols[m] for m in MODS] + [allc]:
        X = D.X[:, cols]
        ps = [predict_flat(fit_flat(X[tr][itr], Ytr[itr], X[tr][iva], Ytr[iva], seed=sid + s), X[q]) for s in range(n_seeds)]
        L.append(np.log(np.clip(np.mean(ps, 0), 1e-9, 1)))
    L.append(np.log(np.clip(knn_geo(D, tr, Ytr, q)[0], 1e-9, 1)))
    return np.stack(L, 1).astype(np.float32)


def ctx_of(D, E, tr, q):
    c = support_context(D, tr, q, E)
    return np.c_[c["dist_km"], c["dens250"], c["env_nov"]].astype(np.float32)


def inner_sets(D, E, tr, Ytr, sid, regime):
    """Inner OOF log-probs + raw support context. regime: 'block' or 'mixed' (block + leave-training-region-out)."""
    b = (D.rc[tr, 0] // 200).astype(np.int64) * 100 + D.rc[tr, 1] // 200
    ub = np.unique(b)
    fb = dict(zip(ub, np.random.default_rng(sid).permutation(len(ub)) % 4))
    f = np.array([fb[x] for x in b])
    parts = [(f != j, f == j) for j in range(4)]
    if regime == "mixed":
        for r in np.unique(D.region[tr]):
            h = D.region[tr] == r
            if h.sum() >= 30 and (~h).sum() >= 200:
                parts.append((~h, h))
    Ls, Cs, Ys, tags = [], [], [], []
    for k, (a, h) in enumerate(parts):
        Ls.append(expert_logp(D, tr[a], Ytr[a], tr[h], sid + 13 * k))
        Cs.append(ctx_of(D, E, tr[a], tr[h]))
        Ys.append(Ytr[h])
        tags.append(np.full(h.sum(), k))
    return np.concatenate(Ls), np.concatenate(Cs), np.concatenate(Ys), np.concatenate(tags)


class Decay(torch.nn.Module):
    def __init__(self, M):
        super().__init__()
        self.raw_w = torch.nn.Parameter(torch.full((M,), -1.0))
        self.log_rho = torch.nn.Parameter(torch.tensor(np.log(200.0), dtype=torch.float32))
        self.raw_beta = torch.nn.Parameter(torch.full((M - 1,), -2.0))
        self.b = torch.nn.Parameter(torch.zeros(5))

    def weights(self, C):
        d, nov = C[:, 0:1], C[:, 2:3]
        kg = torch.exp(-d / torch.exp(self.log_rho))
        ke = torch.exp(-F.softplus(self.raw_beta)[None] * torch.clamp(nov - 1.0, min=0))
        k = torch.cat([ke, kg], 1)
        return F.softplus(self.raw_w)[None] * k

    def forward(self, L, C, logprior):
        w = self.weights(C)
        rest = torch.clamp(1.0 - w.sum(1, keepdim=True), min=0)
        return (L * w[..., None]).sum(1) + rest * logprior[None] + self.b, w


def fit_decay(L, C, Y, logprior, epochs=800):
    torch.manual_seed(0)
    m = Decay(L.shape[1]).to(DEV)
    Lt, Ct, Yt = (torch.as_tensor(a, device=DEV) for a in (L, C, Y))
    lp = torch.as_tensor(logprior, device=DEV, dtype=torch.float32)
    opt = torch.optim.Adam(m.parameters(), lr=0.03)
    for _ in range(epochs):
        opt.zero_grad()
        z, _ = m(Lt, Ct, lp)
        (-(Yt * F.log_softmax(z, 1)).sum() / Yt.sum()).backward()
        opt.step()
    return m.eval()


def decay_apply(m, L, C, logprior):
    with torch.no_grad():
        z, w = m(torch.as_tensor(L, device=DEV), torch.as_tensor(C, device=DEV), torch.as_tensor(logprior, device=DEV, dtype=torch.float32))
    z = z.cpu().numpy()
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True), w.cpu().numpy()


def main(dlist=("block10", "loro", "env", "zone")):
    OUT.mkdir(parents=True, exist_ok=True)
    D = Data()
    DS = designs(D)
    E = env_space(D)
    rows, prm = [], []
    t0 = time.time()
    M = len(EXPERTS)
    for dname in dlist:
        for si, (tr, Ytr, te, Yte, label) in enumerate(DS[dname]):
            sid = 8000 + 37 * si + 11 * len(dname)
            Lte = expert_logp(D, tr, Ytr, te, sid, n_seeds=2)
            Cte = ctx_of(D, E, tr, te)
            logprior = np.log(Ytr.sum(0) / Ytr.sum())
            inner = {rg: inner_sets(D, E, tr, Ytr, sid, rg) for rg in ("block", "mixed")}
            preds = {"Equal": pool(Lte, np.full((1, M, 1), 1.0 / M)), "Only:All (MLP)": pool(Lte, np.eye(M)[EXPERTS.index("All")][None, :, None])}
            for rg, (Lin, Cin, Yin, tag) in inner.items():
                sg = fit_stack(Lin, Yin)
                preds[f"Stack [{rg}]"] = pool(Lte, sg["w"][None], sg["b"])
                mu, sd = np.log1p(Cin).mean(0), np.log1p(Cin).std(0) + 1e-6
                cs_in, cs_te = (np.log1p(Cin) - mu) / sd, (np.log1p(Cte) - mu) / sd
                val_flag = np.where(np.random.default_rng(sid).random(len(tag)) < 0.2, 0, 1)      # 0 = gate early-stopping set
                g = fit_gate(Lin, Yin, cs_in.astype(np.float32), val_flag, seed=sid)
                preds[f"Attn-support [{rg}]"], _ = gate_apply(g, Lte, cs_te.astype(np.float32))
                dm = fit_decay(Lin, Cin, Yin, logprior)
                preds[f"Decay [{rg}]"], wte = decay_apply(dm, Lte, Cte, logprior)
                prm.append({"design": dname, "split": label, "regime": rg, "rho_km": float(np.exp(dm.log_rho.item())),
                            **{f"w_{EXPERTS[m]}": float(F.softplus(dm.raw_w[m]).item()) for m in range(M)},
                            **{f"beta_{EXPERTS[m]}": float(F.softplus(dm.raw_beta[m]).item()) for m in range(M - 1)},
                            "mean_w_geo_test": float(wte[:, -1].mean()), "mean_prior_share_test": float(np.clip(1 - wte.sum(1), 0, None).mean())})
            for nm, P in preds.items():
                rows.append({"design": dname, "split": label, "strategy": nm, **metrics(P, Yte)})
            pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
            pd.DataFrame(prm).to_csv(OUT / "decay_params.csv", index=False)
            print(dname, label, f"{time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    with pd.option_context("display.width", 250):
        print(res.groupby(["strategy", "design"])[["AUC_mean", "nll"]].mean().unstack("design").round(3))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("block10", "loro", "env", "zone"))
