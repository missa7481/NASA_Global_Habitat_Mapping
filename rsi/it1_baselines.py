#!/usr/bin/env python3
"""RSI iteration 1: strong baselines under nine holdout designs, with per-cell out-of-fold predictions.

Models (all on the same fused 105-feature representation unless stated):
  Prior        training class frequencies (floor)
  kNN-geo      distance-weighted composition of 30 nearest training report cells (pure spatial)
  GLM          multinomial logistic (linear early fusion)
  MLP          2x64 MLP, early fusion
  MLP+Coords   MLP + sphere-aware coordinate encoding (coordinate-as-covariate spatial ML)
  HGB          histogram gradient boosting (record-weighted multiclass)
  RF           random forest (record-weighted multiclass)
  GatedMoE     per-class gated mixture of modality experts (HiCo-Fuse)
  HiCo         hierarchically coherent model (HiCo-Fuse)
Neural models: 2 seeds, probabilities averaged.  Outputs: artifacts/rsi/it1/{metrics.csv, oof_<design>.npz}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.common import (ART, Data, designs, fit_flat, fit_net, fit_sklearn, inner_split, knn_geo, metrics,  # noqa: E402
                        predict_flat, predict_net, predict_sklearn)

OUT = ART / "it1"
MODELS = ["Prior", "kNN-geo", "GLM", "MLP", "MLP+Coords", "HGB", "RF", "GatedMoE", "HiCo"]
SEEDS = (0, 1)


def run_split(D, C, tr, Ytr, te, Yte, sid):
    itr, iva = inner_split(D, tr, seed=sid)
    preds = {}
    g = Ytr.sum(0) / Ytr.sum()
    preds["Prior"] = np.tile(g, (len(te), 1))
    preds["kNN-geo"], _ = knn_geo(D, tr, Ytr, te)
    for name, X, lin in (("GLM", D.X, True), ("MLP", D.X, False), ("MLP+Coords", np.c_[D.X, C], False)):
        ps = [predict_flat(fit_flat(X[tr][itr], Ytr[itr], X[tr][iva], Ytr[iva], seed=s + sid, linear=lin), X[te]) for s in SEEDS]
        preds[name] = np.mean(ps, 0)
    for name in ("HGB", "RF"):
        preds[name] = predict_sklearn(fit_sklearn(name, D.X[tr], Ytr, seed=sid), D.X[te])
    for name in ("GatedMoE", "HiCo"):
        ps = [predict_net(fit_net(name, D, tr[itr], Ytr[itr], tr[iva], Ytr[iva], seed=s + sid), D, te, Yte) for s in SEEDS]
        preds[name] = np.mean(ps, 0)
    return preds


def main(only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    D = Data()
    C = D.coords()
    DS = designs(D)
    rows = []
    t0 = time.time()
    for dname, splits in DS.items():
        if only and dname not in only:
            continue
        oof = {m: [] for m in MODELS}
        idx, lab, Ys = [], [], []
        for si, (tr, Ytr, te, Yte, label) in enumerate(splits):
            preds = run_split(D, C, tr, Ytr, te, Yte, sid=1000 + 17 * si)
            for m in MODELS:
                rows.append({"design": dname, "split": label, "model": m, "n_train": len(tr), **metrics(preds[m], Yte)})
                oof[m].append(preds[m].astype(np.float32))
            idx.append(te)
            lab += [label] * len(te)
            Ys.append(Yte)
            print(f"{dname} {label} ntr={len(tr)} nte={len(te)} {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(OUT / f"oof_{dname}.npz", idx=np.concatenate(idx), label=np.array(lab), Y=np.concatenate(Ys),
                            **{m: np.concatenate(v) for m, v in oof.items()})
        pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "metrics.csv", index=False)
    piv = res.groupby(["design", "model"])["AUC_mean"].mean().unstack("design")
    with pd.option_context("display.width", 250):
        print(piv.round(3))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
