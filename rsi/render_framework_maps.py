#!/usr/bin/env python3
"""Dark-theme HiCo-Fuse figures for the public site.

Writes ../github_site/assets/hico/*.webp:
  global_<layer>.webp   0.2-degree views of the four vectors, three disease indices and community support
  zoom_<region>_<level>.webp   coherent global -> regional (0.5 deg) -> local (0.05 deg) -> city windows
  gates_<vector>.webp   dominant local modality gate (HiCo + gated experts)
Rendered images only; no per-cell values are written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hicofuse.postprocess import pct  # noqa: E402
from hicofuse.render import ROW_S60, H, W, block_mean, canvas, hex2rgb, ramp  # noqa: E402

ART = ROOT / "artifacts/hicofuse"
G, R = ART / "global", ART / "results"
OUT = ROOT.parent / "github_site/assets/hico"
NIGHT = ["#0f2238", "#123a63", "#15568c", "#1874b0", "#2b95c8", "#4fb6d6", "#86d3dc", "#c4ead9", "#fff4c2"]
EMBER = ["#1f1626", "#3d1b33", "#62203a", "#8c2a37", "#b8412c", "#db6a22", "#f0a032", "#f8cf6a", "#fff4c2"]
NODATA = (38, 48, 64, 255)
CAT_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]       # climate anomaly, landscape, human, AlphaEarth
ZOOMS = {
    "florida": {"cls": 0, "box": (24.0, 31.5, -88.0, -79.5), "city": (25.3, 26.4, -80.9, -79.9)},
    "seasia": {"cls": 1, "box": (-1.0, 8.0, 98.0, 106.0), "city": (1.1, 1.6, 103.55, 104.1)},
    "italy": {"cls": 0, "box": (40.0, 46.5, 7.0, 14.5), "city": (41.7, 42.1, 12.25, 12.75)},
}


def save(rgba, name):
    Image.fromarray(rgba, "RGBA").save(OUT / f"{name}.webp", quality=88, method=6)
    print(name, flush=True)


def paint(g, land, stops, gamma=1.0):
    rgba = np.zeros(g.shape + (4,), np.uint8)
    ok = np.isfinite(g)
    rgba[land & ~ok] = NODATA
    rgba[ok, :3] = ramp(np.clip(g[ok], 0, 1) ** gamma, stops).astype(np.uint8)
    rgba[ok, 3] = 255
    return rgba


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rc = np.load(G / "rc.npy")
    store_rc = np.load(ART / "store/rc.npy")
    land = np.zeros((H, W), bool)
    land[store_rc[:, 0], store_rc[:, 1]] = True
    k = 4
    land4 = block_mean(land[:ROW_S60].astype(np.float32), k) > 0.3

    L = np.load(R / "layers.npz")
    for key, name in (("suit_Ae_albopictus", "albopictus"), ("suit_Ae_aegypti", "aegypti"),
                      ("suit_Culex", "culex"), ("suit_Anopheles", "anopheles")):
        save(paint(block_mean(canvas(L[key], rc)[:ROW_S60], k), land4, NIGHT, 1.6), f"global_{name}")
    for name in ("dengue", "wnv", "malaria"):
        save(paint(block_mean(canvas(pct(L[name]).astype(np.float32), rc)[:ROW_S60], k), land4, EMBER, 2.2), f"global_{name}")
    save(paint(block_mean(canvas(pct(np.load(G / "support.npy")).astype(np.float32), rc)[:ROW_S60], k), land4, NIGHT, 1.6),
         "global_support")

    # coherent zoom: coarse parent mean and fine children on one shared square-root scale
    tp = np.load(G / ("HF2_twopart.npy" if (G / "HF2_twopart.npy").exists() else "HiCo_twopart.npy"))
    par = np.load(G / "parent.npy")
    n_par = par.max() + 1
    for zn, z in ZOOMS.items():
        v = tp[:, z["cls"]].astype(np.float64)
        pm = np.bincount(par, weights=v, minlength=n_par) / np.bincount(par, minlength=n_par)
        fine, coarse = canvas(v, rc), canvas(pm[par], rc)
        lat0, lat1, lon0, lon1 = z["box"]
        r0, r1, c0, c1 = int((85 - lat1) / .05), int((85 - lat0) / .05), int((lon0 + 180) / .05), int((lon1 + 180) / .05)
        fw, cw, lm = fine[r0:r1, c0:c1], coarse[r0:r1, c0:c1], land[r0:r1, c0:c1]
        lo, hi = np.nanpercentile(fw, 2), np.nanpercentile(fw, 99.5)
        sq = lambda a: np.sqrt(np.clip((a - lo) / (hi - lo), 0, 1))  # noqa: E731
        up = lambda a, n: np.kron(a, np.ones((n, n)))  # noqa: E731
        save(paint(up(sq(cw), 4), up(lm.astype(float), 4) > 0, NIGHT), f"zoom_{zn}_coarse")
        save(paint(up(sq(fw), 4), up(lm.astype(float), 4) > 0, NIGHT), f"zoom_{zn}_fine")
        a0, a1, b0, b1 = z["city"]
        rr0, rr1, cc0, cc1 = int((85 - a1) / .05), int((85 - a0) / .05), int((b0 + 180) / .05), int((b1 + 180) / .05)
        save(paint(up(sq(fine[rr0:rr1, cc0:cc1]), 24), up(land[rr0:rr1, cc0:cc1].astype(float), 24) > 0, NIGHT), f"zoom_{zn}_city")

    gates = np.load(G / "HiCoMoE_gates.npy", mmap_mode="r")
    for s, nm in ((0, "albopictus"), (1, "aegypti")):
        dom = np.asarray(gates[:, :, s], dtype=np.float32).argmax(1)
        c = np.full((H, W), -1, np.int16)
        c[rc[:, 0], rc[:, 1]] = dom
        c = c[:ROW_S60:k, ::k]
        rgba = np.zeros(c.shape + (4,), np.uint8)
        rgba[land4 & (c < 0)] = NODATA
        for m in range(4):
            rgba[c == m, :3] = hex2rgb(CAT_DARK[m]).astype(np.uint8)
            rgba[c == m, 3] = 255
        save(rgba, f"gates_{nm}")


if __name__ == "__main__":
    main()
