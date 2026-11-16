#!/usr/bin/env python3
"""Multi-scale AlphaEarth 2024 means for development + NEON sites (step 3 input).

The frozen recipe uses a 500 m geodesic-buffer mean on a 250 m working grid.
Here the same image graph (g.raw_region 'alphaearth') is reduced at additional
radii: 100 m (10 m native working scale) and 2000 m (250 m working scale), so
that a model can learn the habitat-relevant spatial scale.  The 500 m values
are taken from the frozen features (not re-extracted).  Output is a local
research derivative with site identifiers; do not publish.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import sys
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import global_mapping_2024 as g  # noqa: E402
from extract_new_points import jobs_for  # noqa: E402

SCALES = {100: 10, 2000: 250}  # radius_m -> working scale_m


def records(rows, crs, radius, scale):
    pts = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r["longitude"], r["latitude"]]), {"sample_id": str(r["sample_id"])}) for r in rows])
    pad = 0.02 + radius / 111_000
    region = ee.Geometry.Rectangle([min(r["longitude"] for r in rows) - pad, min(r["latitude"] for r in rows) - pad,
                                    max(r["longitude"] for r in rows) + pad, max(r["latitude"] for r in rows) + pad], geodesic=False)
    im = g.raw_region(region.buffer(radius + 1000, 10), "alphaearth")
    buffers = pts.map(lambda f: ee.Feature(f.geometry().buffer(radius, 1), f.toDictionary()))
    fc = im.reduceRegions(collection=buffers, reducer=ee.Reducer.mean(), crs=crs, scale=scale, tileScale=4)
    return [f["properties"] for f in fc.map(lambda f: ee.Feature(None, f.toDictionary())).getInfo()["features"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "oadsdm" / "multiscale")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    cache = a.out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    loc = pd.read_pickle(g.O / "point_features.pkl")[["longitude", "latitude"]].reset_index().rename(columns={"index": "sample_id"})
    loc = loc.rename(columns={loc.columns[0]: "sample_id"})
    g.init()
    jobs = jobs_for(loc)
    # split large tiles so a 10 m reduction stays within request limits
    small = []
    for j in jobs:
        for k in range(0, len(j["rows"]), 200):
            small.append({**j, "id": f"{j['id']}_{k // 200}", "rows": j["rows"][k:k + 200]})
    tasks = [(j, r) for j in small for r in SCALES]

    def one(task):
        j, r = task
        dest = cache / f"{j['id']}_r{r}.json.gz"
        if dest.exists():
            return "cached"
        for attempt in range(5):
            try:
                rec = records(j["rows"], j["crs"], r, SCALES[r])
                assert len(rec) == len(j["rows"])
                with gzip.open(dest, "wt", encoding="utf-8") as f:
                    json.dump(rec, f, allow_nan=False)
                return "downloaded"
            except Exception as e:  # noqa: BLE001
                err = str(e)[:200]
                time.sleep([5, 15, 45, 90, 180][attempt])
        return "failed: " + err

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        states = list(pool.map(one, tasks))
    fails = [s for s in states if s.startswith("failed")]
    print(len(tasks), "tasks", len(fails), "failed", f"{time.time() - t0:.0f}s", fails[:3], flush=True)
    if fails:
        return
    out = {}
    for r in SCALES:
        frames = [pd.DataFrame(json.load(gzip.open(p, "rt", encoding="utf-8"))) for p in sorted(cache.glob(f"*_r{r}.json.gz"))]
        t = pd.concat(frames, ignore_index=True).set_index("sample_id")
        out[r] = t[g.AE + ["alpha_coverage"]].add_suffix(f"_r{r}")
    feats = pd.concat(out.values(), axis=1)
    feats.to_pickle(a.out / "alphaearth_multiscale.pkl")
    print("multiscale", feats.shape, "complete", int(feats.notna().all(axis=1).sum()))


if __name__ == "__main__":
    main()
