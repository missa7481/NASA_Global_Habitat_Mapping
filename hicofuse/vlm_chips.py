#!/usr/bin/env python3
"""Download Sentinel-2 2024 median RGB chips (one per 0.05-degree cell) for the VLM concept module.

Sample: class-stratified report cells + random land cells (for concept distillation diversity).
Chip = the full 0.05-degree cell footprint, 224 x 224 px, S2 SR harmonized, 2024, scenes with
<20% cloud, median composite, RGB stretch 0-3000. Local research cache; not for redistribution
beyond Copernicus terms (free and open, attribution required).
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import ee
import numpy as np

ART = Path(__file__).resolve().parents[1] / "artifacts/hicofuse"
OUT = ART / "vlm/chips"


def main(n_random=600, cap=350, workers=12):
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(ART / "dataset.npz")
    rc_all, obs, Y, complete = d["rc"], d["obs_idx"], d["Y"], d["complete"]
    rng = np.random.default_rng(3)
    pick = []
    for s in range(5):
        cells = obs[Y[:, s] > 0]
        pick += list(rng.choice(cells, min(cap, len(cells)), replace=False))
    land = np.flatnonzero(complete)
    pick += list(rng.choice(land, n_random, replace=False))
    pick = np.unique(pick)
    (ART / "vlm").mkdir(exist_ok=True)
    np.save(ART / "vlm/chip_store_rows.npy", pick)
    ee.Initialize(project=os.environ["EE_PROJECT"])   # your Google Cloud project with Earth Engine enabled
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate("2024-01-01", "2025-01-01")
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).select(["B4", "B3", "B2"]))

    def one(i):
        dest = OUT / f"{i}.png"
        if dest.exists():
            return "cached"
        r, c = rc_all[i]
        lat0, lon0 = 85 - (r + 1) * 0.05, -180 + c * 0.05
        region = ee.Geometry.Rectangle([lon0, lat0, lon0 + 0.05, lat0 + 0.05], geodesic=False)
        img = col.filterBounds(region).median().visualize(min=0, max=3000, bands=["B4", "B3", "B2"])
        for a in range(4):
            try:
                url = img.getThumbURL({"region": region, "dimensions": 224, "format": "png"})
                with urllib.request.urlopen(url, timeout=120) as resp:
                    dest.write_bytes(resp.read())
                return "ok"
            except Exception as e:  # noqa: BLE001
                err = str(e)[:120]
                time.sleep(3 * (a + 1))
        return "fail " + err

    t0 = time.time()
    states = []
    order = np.random.default_rng(9).permutation(pick)   # random order: any partial download is representative
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for k, s in enumerate(pool.map(one, order)):
            states.append(s)
            if k % 200 == 0:
                print(k, len(pick), s, f"{time.time() - t0:.0f}s", flush=True)
    summary = {s.split()[0]: states.count(s) for s in set(states)}
    (ART / "vlm/chips_manifest.json").write_text(json.dumps({"n": len(pick), "states": summary}, indent=1))
    print(summary)


if __name__ == "__main__":
    main()
