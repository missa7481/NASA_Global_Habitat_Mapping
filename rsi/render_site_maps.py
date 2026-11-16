#!/usr/bin/env python3
"""Dark-theme global maps for the public site (0.2-degree view, transparent ocean)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hicofuse.render import ROW_S60, H, W, block_mean, canvas, ramp  # noqa: E402

OUT = ROOT / "artifacts/rsi/site_preview/assets"
OUT.mkdir(parents=True, exist_ok=True)
# monotone in lightness: deep navy -> cyan -> pale gold (reads on a near-black page)
NIGHT = ["#0f2238", "#123a63", "#15568c", "#1874b0", "#2b95c8", "#4fb6d6", "#86d3dc", "#c4ead9", "#fff4c2"]
NODATA = (38, 48, 64, 255)


def main():
    G, R = ROOT / "artifacts/hicofuse/global", ROOT / "artifacts/hicofuse/results"
    rc = np.load(G / "rc.npy")
    store_rc = np.load(ROOT / "artifacts/hicofuse/store/rc.npy")
    land = np.zeros((H, W), bool)
    land[store_rc[:, 0], store_rc[:, 1]] = True
    k = 4
    land4 = block_mean(land[:ROW_S60].astype(np.float32), k) > 0.3
    L = np.load(R / "layers.npz")
    for key, name in (("suit_Ae_albopictus", "albopictus"), ("suit_Ae_aegypti", "aegypti"),
                      ("suit_Culex", "culex"), ("suit_Anopheles", "anopheles")):
        g = block_mean(canvas(L[key], rc)[:ROW_S60], k)
        rgba = np.zeros(g.shape + (4,), np.uint8)
        ok = np.isfinite(g)
        rgba[land4 & ~ok] = NODATA
        rgba[ok, :3] = ramp(np.clip(g[ok], 0, 1) ** 1.6, NIGHT).astype(np.uint8)   # emphasise the upper tail
        rgba[ok, 3] = 255
        Image.fromarray(rgba, "RGBA").save(OUT / f"map_{name}.png", optimize=True)
        print(name, flush=True)


if __name__ == "__main__":
    main()
