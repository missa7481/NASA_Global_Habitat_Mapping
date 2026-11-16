#!/usr/bin/env python3
"""Multi-resolution fusion experiment: does coarse-level evidence help fine-level prediction?

Evidence: iNaturalist reports at 0.05 deg (fine, composition likelihood) +
Kraemer et al. (2015) compendium for Ae. aegypti / Ae. albopictus, used at 0.5 deg
(parent presence vs all land parents, logistic presence-background with own intercept).

Arms (identical folds/optimiser/early stopping):
  MLPEarly          fine iNat only
  MLPEarly+Kfine    + Kraemer cells pooled as 0.05-deg presences vs 50k random land cells (naive)
  HiCo              fine iNat only, coherent hierarchy
  HiCo+Kcoarse      + Kraemer at parent level supervising the macro term A (resolution-aware)
Evaluation: held-out 10-deg blocks; all test cells and the tropical subset (|lat| < 23.44).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicofuse.models import Net, multinomial_nll  # noqa: E402
from hicofuse.train import ART, DEV, Problem, Store, class_metrics, split_inner  # noqa: E402

ARMS = ["MLPEarly", "MLPEarly+Kfine", "HiCo", "HiCo+Kcoarse"]


def kraemer_targets(st: Store):
    d = st.d
    zK = np.zeros((len(st.par_keys), 2), np.float32)        # species 0 albopictus, 1 aegypti
    fine = {}
    for s, sp in ((0, "albopictus"), (1, "aegypti")):
        p = st.pos[d[f"kraemer_{sp}"]]
        p = p[p >= 0]
        zK[np.unique(st.par[p]), s] = 1
        fine[s] = p
    return zK, fine


def fit_arm(arm, st, train, val, seed, aux, max_epochs=800, patience=60):
    torch.manual_seed(seed)
    base = arm.split("+")[0]
    net = Net(base, {m: len(c) for m, c in st.cols.items()}).to(DEV)
    cK = nn.Parameter(torch.zeros(2, device=DEV))
    opt = torch.optim.AdamW(list(net.parameters()) + [cK], lr=3e-3, weight_decay=1e-4)
    best, state, bad = np.inf, None, 0
    for ep in range(max_epochs):
        net.train()
        opt.zero_grad()
        loss = multinomial_nll(train.eta(net), train.Y)
        if arm.endswith("Kcoarse"):
            A = net.A(aux["par_all"])[:, :2] + cK
            loss = loss + F.binary_cross_entropy_with_logits(A, aux["zK"])
        elif arm.endswith("Kfine"):
            e = net.flat(aux["fine_b"])[:, :2]
            idx = aux["fine_s"]
            logit = e[torch.arange(len(idx), device=DEV), idx] + cK[idx]
            loss = loss + F.binary_cross_entropy_with_logits(logit, aux["fine_y"])
        loss.backward()
        opt.step()
        if ep % 5 == 4:
            net.eval()
            with torch.no_grad():
                v = float(multinomial_nll(val.eta(net), val.Y))   # selection on fine iNat validation only
            if v < best - 1e-4:
                best, bad = v, 0
                state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 5
                if bad >= patience:
                    break
    net.load_state_dict(state)
    net.eval()
    return net


def main(seeds=(0, 1)):
    st = Store()
    d = st.d
    obs_pos = st.pos[d["obs_idx"]]
    Y, fold = d["Y"], d["fold"]
    blocks = (st.rc[obs_pos, 0] // 200).astype(np.int64) * 100 + st.rc[obs_pos, 1] // 200
    lat = 85 - (st.rc[obs_pos, 0] + 0.5) * 0.05
    zK, fine = kraemer_targets(st)
    rng = np.random.default_rng(11)
    bg = rng.choice(len(st.cell), 50_000, replace=False)
    fine_pos = np.r_[fine[0], fine[1], bg, bg]
    fine_s = np.r_[np.zeros(len(fine[0])), np.ones(len(fine[1])), np.zeros(len(bg)), np.ones(len(bg))].astype(np.int64)
    fine_y = np.r_[np.ones(len(fine[0]) + len(fine[1])), np.zeros(2 * len(bg))].astype(np.float32)
    aux = {"par_all": torch.as_tensor(st.par_all, device=DEV), "zK": torch.as_tensor(zK, device=DEV),
           "fine_b": st.batch(fine_pos), "fine_s": torch.as_tensor(fine_s, device=DEV),
           "fine_y": torch.as_tensor(fine_y, device=DEV)}
    rows, t0 = [], time.time()
    for k in range(5):
        te = fold == k
        test = Problem(st, obs_pos[te], Y[te])
        trop = np.abs(lat[te]) < 23.44
        for seed in seeds:
            trp, vap, _ = split_inner(st, np.flatnonzero(~te), blocks[~te], 100 * k + seed)
            train, val = Problem(st, obs_pos[trp], Y[trp]), Problem(st, obs_pos[vap], Y[vap])
            for arm in ARMS:
                net = fit_arm(arm, st, train, val, 1000 * k + seed, aux)
                with torch.no_grad():
                    e = test.eta(net)
                    nll = float(multinomial_nll(e, test.Y))
                e = e.cpu().numpy()
                row = {"fold": k, "seed": seed, "arm": arm, "test_nll": nll, **class_metrics(e, Y[te])}
                if trop.sum() > 20:
                    row.update({f"trop_{a}": b for a, b in class_metrics(e[trop], Y[te][trop]).items()})
                    row["n_trop"] = int(trop.sum())
                rows.append(row)
            print(f"fold {k} seed {seed} {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ART / "multires_metrics.csv", index=False)
    cols = ["test_nll", "AUC_Ae_albopictus", "AUC_Ae_aegypti", "AP_Ae_albopictus", "AP_Ae_aegypti",
            "trop_AUC_Ae_albopictus", "trop_AUC_Ae_aegypti"]
    summ = res.groupby("arm")[[c for c in cols if c in res]].mean()
    summ.to_csv(ART / "multires_summary.csv")
    with pd.option_context("display.width", 250):
        print(summ.round(3))
        print(res.groupby("arm").n_trop.sum() / len(seeds))


if __name__ == "__main__":
    main()
