#!/usr/bin/env python3
"""Mosquito-community habitat (support) term h(x) for the two-part map.

  logit P(cell has any Culicidae report) = h(Climate, Landscape) + gamma . Human + c
Positives: all 7,004 report cells; background: 60,000 random complete land cells.
The linear Human term (population, built-up, night lights, travel time) absorbs observer
access; for mapping it is held at its standardized global mean (0), the bias-correction
approach of Warton et al. (2013). Because human footprint is also genuine habitat for
peridomestic vectors, that part is carried by the species-share term, not by h.
Species intensity map:  S_s(x) = sigmoid(h(x) + c) * pi_s(x),  pi_s = softmax share from HiCo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicofuse.models import mlp  # noqa: E402
from hicofuse.train import ART, DEV, Store  # noqa: E402


class Support(nn.Module):
    def __init__(self, d_env, d_h):
        super().__init__()
        self.h = mlp(d_env, 1, hidden=64)
        self.g = nn.Linear(d_h, 1)

    def forward(self, env, hum, effort=True):
        z = self.h(env).squeeze(-1)
        return z + self.g(hum).squeeze(-1) if effort else z + self.g.bias


def tensors(st, pos):
    x = st.X[pos]
    env = np.c_[x[:, st.cols["Climate"]], x[:, st.cols["Landscape"]]]
    return torch.as_tensor(env, device=DEV), torch.as_tensor(x[:, st.cols["Human"]], device=DEV)


def fit(st, pos, y, seed=0, epochs=600):
    torch.manual_seed(seed)
    env, hum = tensors(st, pos)
    yt = torch.as_tensor(y, dtype=torch.float32, device=DEV)
    m = Support(env.shape[1], hum.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(m(env, hum), yt).backward()
        opt.step()
    return m.eval()


def main():
    st = Store()
    d = st.d
    obs = st.pos[d["obs_idx"]]
    rng = np.random.default_rng(21)
    bg = np.setdiff1d(rng.choice(len(st.cell), 60_000, replace=False), obs)
    pos = np.r_[obs, bg]
    y = np.r_[np.ones(len(obs)), np.zeros(len(bg))]
    blk = (st.rc[pos, 0] // 200).astype(np.int64) * 100 + st.rc[pos, 1] // 200
    ub = np.unique(blk)
    fold_of = dict(zip(ub, np.random.default_rng(4).permutation(len(ub)) % 5))
    fold = np.array([fold_of[b] for b in blk])
    a_full, a_corr = [], []
    for k in range(5):
        tr, te = fold != k, fold == k
        m = fit(st, pos[tr], y[tr], seed=k)
        env, hum = tensors(st, pos[te])
        with torch.no_grad():
            a_full.append(roc_auc_score(y[te], m(env, hum).cpu().numpy()))
            a_corr.append(roc_auc_score(y[te], m(env, hum, effort=False).cpu().numpy()))
    print("support block-CV AUC with effort term %.3f, effort held at mean %.3f" % (np.mean(a_full), np.mean(a_corr)))
    m = fit(st, pos, y, seed=0)
    out = []
    with torch.no_grad():
        for i in range(0, len(st.cell), 500_000):
            env, hum = tensors(st, np.arange(i, min(i + 500_000, len(st.cell))))
            out.append(torch.sigmoid(m(env, hum, effort=False)).cpu().numpy())
    sup = np.concatenate(out).astype(np.float32)
    np.save(ART / "global" / "support.npy", sup)
    np.save(ART / "results" / "support_cv.npy", np.array([np.mean(a_full), np.mean(a_corr)]))
    gamma = m.g.weight.detach().cpu().numpy().ravel()
    print("effort coefficients (pop, built, lights, access):", np.round(gamma, 3))


if __name__ == "__main__":
    main()
