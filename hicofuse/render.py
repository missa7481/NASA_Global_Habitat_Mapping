#!/usr/bin/env python3
"""Render HiCo-Fuse map figures (PNG, transparent ocean) + JSON data for report charts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ART = Path(__file__).resolve().parents[1] / "artifacts/hicofuse"
G, R = ART / "global", ART / "results"
FIG = ART / "report"
H, W = 3400, 7200
ROW_S60 = 2900                                   # crop at 60 S

SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIV = ["#1c5cab", "#3987e5", "#86b6ef", "#cde2fb", "#f0efec", "#f6c7c3", "#ee9391", "#e34948", "#b8302f"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]           # categorical slots 1-4
NODATA_LAND = (205, 203, 196, 255)


def hex2rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


def ramp(vals, stops):
    cols = np.array([hex2rgb(h) for h in stops])
    v = np.clip(vals, 0, 1) * (len(stops) - 1)
    i = np.clip(np.floor(v).astype(int), 0, len(stops) - 2)
    f = (v - i)[..., None]
    return cols[i] * (1 - f) + cols[i + 1] * f


def canvas(values, rc, fill=np.nan):
    c = np.full((H, W), fill, np.float32)
    c[rc[:, 0], rc[:, 1]] = values
    return c


def block_mean(a, k):
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    b = a[:h, :w].reshape(h // k, k, w // k, k)
    with np.errstate(invalid="ignore"):
        return np.nanmean(b, axis=(1, 3))


def to_png(grid, path, stops=SEQ, landmask=None, vmin=0.0, vmax=1.0):
    rgba = np.zeros(grid.shape + (4,), np.uint8)
    ok = np.isfinite(grid)
    if landmask is not None:
        rgba[landmask & ~ok] = NODATA_LAND
    rgba[ok, :3] = ramp((grid[ok] - vmin) / (vmax - vmin), stops).astype(np.uint8)
    rgba[ok, 3] = 255
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rc = np.load(G / "rc.npy")
    store_rc = np.load(ART / "store/rc.npy")
    land = np.zeros((H, W), bool)
    land[store_rc[:, 0], store_rc[:, 1]] = True
    L = np.load(R / "layers.npz")
    k = 4                                                           # 0.2-degree global view
    land4 = block_mean(land[:ROW_S60].astype(np.float32), k) > 0.3
    layers = {"albopictus": L["suit_Ae_albopictus"], "aegypti": L["suit_Ae_aegypti"], "culex": L["suit_Culex"],
              "anopheles": L["suit_Anopheles"], "dengue": None, "wnv": None, "malaria": None}
    from hicofuse.postprocess import pct  # noqa: E402
    for name in ("dengue", "wnv", "malaria"):
        layers[name] = pct(L[name]).astype(np.float32)
    layers["support"] = pct(np.load(G / "support.npy")).astype(np.float32)
    for name, v in layers.items():
        if (FIG / f"global_{name}.png").exists():
            continue
        g = block_mean(canvas(v, rc)[:ROW_S60], k)
        to_png(g, FIG / f"global_{name}.png", landmask=land4)
        print("global", name, flush=True)

    # ---- global -> regional -> local coherent zoom (raw two-part intensity, shared scale) ----
    tp = np.load(G / "HiCo_twopart.npy")
    par = np.load(G / "parent.npy")
    n_par = par.max() + 1
    zooms = {
        "florida": {"cls": 0, "box": (24.0, 31.5, -88.0, -79.5), "city": (25.3, 26.4, -80.9, -79.9), "label": "Florida, USA - Ae. albopictus"},
        "seasia": {"cls": 1, "box": (-1.0, 8.0, 98.0, 106.0), "city": (1.1, 1.6, 103.55, 104.1), "label": "Malay Peninsula - Ae. aegypti"},
        "italy": {"cls": 0, "box": (40.0, 46.5, 7.0, 14.5), "city": (41.7, 42.1, 12.25, 12.75), "label": "Northern Italy - Ae. albopictus"},
    }
    zmeta = {}
    for zn, z in zooms.items():
        s = z["cls"]
        v = tp[:, s].astype(np.float64)
        pm = np.bincount(par, weights=v, minlength=n_par) / np.bincount(par, minlength=n_par)
        fine = canvas(v, rc)
        coarse = canvas(pm[par], rc)
        lat0, lat1, lon0, lon1 = z["box"]
        r0, r1 = int((85 - lat1) / 0.05), int((85 - lat0) / 0.05)
        c0, c1 = int((lon0 + 180) / 0.05), int((lon1 + 180) / 0.05)
        fwin, cwin = fine[r0:r1, c0:c1], coarse[r0:r1, c0:c1]
        lo, hi = np.nanpercentile(fwin, 2), np.nanpercentile(fwin, 99.5)
        lm = land[r0:r1, c0:c1]
        sq = lambda a: np.sqrt(np.clip((a - lo) / (hi - lo), 0, 1))  # noqa: E731  same monotone scale both levels
        up = lambda a: np.kron(a, np.ones((4, 4)))  # noqa: E731
        to_png(up(sq(cwin)), FIG / f"zoom_{zn}_coarse.png", landmask=up(lm.astype(float)) > 0)
        to_png(up(sq(fwin)), FIG / f"zoom_{zn}_fine.png", landmask=up(lm.astype(float)) > 0)
        a0, a1, b0, b1 = z["city"]
        rr0, rr1 = int((85 - a1) / 0.05), int((85 - a0) / 0.05)
        cc0, cc1 = int((b0 + 180) / 0.05), int((b1 + 180) / 0.05)
        cityf = fine[rr0:rr1, cc0:cc1]
        upc = lambda a: np.kron(a, np.ones((24, 24)))  # noqa: E731
        to_png(upc(sq(cityf)), FIG / f"zoom_{zn}_city.png", landmask=upc(land[rr0:rr1, cc0:cc1].astype(float)) > 0)
        # coherence numbers for this window
        pids = np.unique(par[(rc[:, 0] >= r0) & (rc[:, 0] < r1) & (rc[:, 1] >= c0) & (rc[:, 1] < c1)])
        zmeta[zn] = {**z, "parents": int(len(pids)), "color_range": [float(lo), float(hi)]}
        print("zoom", zn, flush=True)
    (FIG / "zoom_meta.json").write_text(json.dumps(zmeta, indent=1))

    # ---- dominant local modality (HiCoMoE gates, Ae. albopictus and Ae. aegypti) ----
    gates = np.load(G / "HiCoMoE_gates.npy", mmap_mode="r")         # N, M(ClimAnom, Landscape, Human, AE), S
    for s, nm in ((0, "albopictus"), (1, "aegypti")):
        gs = np.asarray(gates[:, :, s], dtype=np.float32)
        dom = gs.argmax(1)
        c = np.full((H, W), -1, np.int16)
        c[rc[:, 0], rc[:, 1]] = dom
        c = c[:ROW_S60:k, ::k]
        rgba = np.zeros(c.shape + (4,), np.uint8)
        rgba[land4 & (c < 0)] = NODATA_LAND
        for m in range(4):
            rgba[c == m, :3] = hex2rgb(CAT[m]).astype(np.uint8)
            rgba[c == m, 3] = 255
        Image.fromarray(rgba, "RGBA").save(FIG / f"gates_dominant_{nm}.png", optimize=True)
        np.save(R / f"gate_share_{nm}.npy", np.bincount(dom, minlength=4) / len(dom))
        print("gates", nm, np.round(gs.mean(0), 3), flush=True)

    # ---- incoherence of a separately fitted coarse model (parent level), diverging ----
    print("done")


if __name__ == "__main__":
    main()
