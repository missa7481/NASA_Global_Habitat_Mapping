#!/usr/bin/env python3
"""Global HiCo-Fuse 2 inference: RF niche term + coherent pool/calibration head + novelty trust, all 0.05-deg land cells.

Same estimator as rsi/it7_hicofuse2.py (HF2-shrink2), fitted on every report cell:
  head  <- 4-fold 5-degree-block OOF RF output on all report cells
  trust <- min(covariate-novelty trust, pool-novelty trust) of each 0.5-deg parent vs the parents with reports
Writes artifacts/hicofuse/global/HF2_eta.npy and HF2_parentA.npy (same layout as HiCo_eta / HiCo_parentA), so
hicofuse/postprocess.py and rsi/render_framework_maps.py consume it unchanged.

  HF2_POOL=all python rsi/hf2_global.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.common import Data, fit_sklearn  # noqa: E402
from rsi.it7_hicofuse2 import POOL, Head, novelty, oof_problem, parent_pool, rf_eta, trust  # noqa: E402
from hicofuse.train import ART  # noqa: E402

G = ART / "global"


def main():
    t0 = time.time()
    D = Data()
    st = D.st
    Zall = parent_pool(st)
    tr = np.arange(len(D.Y))
    Bo, go, Bc, gc, n, pk, q90 = oof_problem(D, tr, D.Y, 0, Zall)
    H = Head(Zall.shape[1], pool=True).fit(Bo, go, Bc, gc, n, Zall[pk], D.Y)
    print("head", H.params(), f"{time.time() - t0:.0f}s", flush=True)
    rf = fit_sklearn("RF", D.X, D.Y, seed=0)
    order = np.load(G / "cell_store_rows.npy")
    assert np.array_equal(order, st.cell), "global arrays must follow the store's complete-cell order"
    B = rf_eta(rf, st.X).astype(np.float64)
    print("RF on", len(B), "cells", f"{time.time() - t0:.0f}s", flush=True)
    n_par = len(st.par_keys)
    allp = np.arange(n_par)
    ref = np.unique(st.par[D.pos])
    rho = np.minimum(trust(novelty(st.par_all, allp, ref), q90[0]), trust(novelty(Zall, allp, ref), q90[1]))
    eta, A = H.predict(B, st.par, B, st.par, n_par, Zall, rho=rho)
    np.save(G / "HF2_eta.npy", eta.astype(np.float32))
    np.save(G / "HF2_parentA.npy", A.astype(np.float32))
    meta = {"model": "HiCo-Fuse 2 (HF2-shrink2)", "pool": POOL, "head": H.params(), "q90": q90,
            "trust_share_full": float((rho == 1).mean()), "trust_share_rf": float((rho == 0).mean()),
            "trust_mean": float(rho.mean()), "parents": int(n_par), "cells": int(len(B))}
    (G / "HF2_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "head"}), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
