#!/usr/bin/env python3
"""Assemble the global 0.05-degree feature store from the frozen deployment tiles.

Output (artifacts/hicofuse/store/):
  X.npy      float32 (n_cells, 105)  BIO19 + ENV22 + AE64, NaN where a product is missing
  rc.npy     int32   (n_cells, 2)    global grid row, col  (lat = 85-(r+.5)*.05, lon = -180+(c+.5)*.05)
  meta.json  feature names, grid definition, source tile count
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TILES = ROOT / "data/derived/global_mapping_2024/deployment/tiles"
OUT = Path(__file__).resolve().parents[1] / "artifacts/hicofuse/store"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feats = json.loads((ROOT / "data/derived/global_mapping_2024/deployment/protocol.json").read_text())["features"]
    files = sorted(glob.glob(str(TILES / "*.npz")))
    Xs, RC = [], []
    for i, f in enumerate(files):
        z = np.load(f)
        job = json.loads(Path(f).with_suffix(".json").read_text())["job"]  # rows/cols are tile-local
        Xs.append(z["features"].astype(np.float32))
        RC.append(np.c_[z["rows"] + job["row"], z["cols"] + job["col"]].astype(np.int32))
        if i % 500 == 0:
            print(i, len(files), flush=True)
    X = np.concatenate(Xs)
    rc = np.concatenate(RC)
    order = np.lexsort((rc[:, 1], rc[:, 0]))
    X, rc = X[order], rc[order]
    assert len(np.unique(rc[:, 0].astype(np.int64) * 10000 + rc[:, 1])) == len(rc), "duplicate cells"
    np.save(OUT / "X.npy", X)
    np.save(OUT / "rc.npy", rc)
    (OUT / "meta.json").write_text(json.dumps({
        "features": feats, "n_cells": int(len(rc)), "tiles": len(files),
        "grid": {"step_deg": 0.05, "lat_of_row": "85-(row+0.5)*0.05", "lon_of_col": "-180+(col+0.5)*0.05"},
        "complete_rows": int(np.isfinite(X).all(1).sum()),
    }, indent=1))
    print("store", X.shape, "complete", int(np.isfinite(X).all(1).sum()))


if __name__ == "__main__":
    main()
