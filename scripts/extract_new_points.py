#!/usr/bin/env python3
"""Extract the frozen 105-feature recipe for NEW point locations (e.g. AIMSurv traps).

Reuses the project's own extraction functions unchanged, so values follow the
same recipe as ``point_features.pkl``:
  * AlphaEarth 2024 + environment: 500 m geodesic-buffer means on a 250 m working grid
  * ERA5-Land: native-pixel value
  * GHSL built fraction: corrected native-area denominator (built amendment)
  * WorldPop 2024: native count summed in the 500 m circle / full circle area (WorldPop amendment)
  * WorldClim BIO19: local native 2.5' rasters, nearest pixel
Writes only under --out (its own cache); never touches the frozen global_mapping_2024 outputs.
``--check-existing N`` re-extracts N already-frozen NEON sites and reports max abs differences.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import global_mapping_2024 as g  # noqa: E402
import global_mapping_built as gb  # noqa: E402
import global_mapping_worldpop as gw  # noqa: E402

TILE = 64 * g.STEP


def jobs_for(loc: pd.DataFrame) -> list[dict]:
    loc = loc.copy()
    loc["tr"] = np.floor((85 - loc.latitude) / TILE).astype(int)
    loc["tc"] = np.floor((loc.longitude + 180) / TILE).astype(int)
    jobs = []
    for (r, c), d in loc.groupby(["tr", "tc"]):
        lon, lat = float(d.longitude.mean()), float(d.latitude.mean())
        jobs.append({"id": f"t{r:03d}_{c:03d}", "crs": g.projection(lon, lat),
                     "rows": d[["sample_id", "longitude", "latitude"]].to_dict("records")})
    return jobs


def fetch(job: dict, kind: str, cache: Path) -> dict:
    dest = cache / f"{job['id']}_{kind}.json.gz"
    if dest.exists():
        return {"job": job["id"], "kind": kind, "status": "cached"}
    last = None
    for attempt in range(5):
        try:
            rows = job["rows"]
            if kind in ("alphaearth", "environment", "era5"):
                rec = g.sparse_records(rows, kind, job["crs"])
            elif kind == "built":
                rec = gb.records(rows, job["crs"])
            elif kind == "worldpop":
                rec = gw.records(rows)
            if len(rec) != len(rows):
                raise ValueError("record count mismatch")
            tmp = dest.with_suffix(".part")
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump(rec, f, allow_nan=False)
            tmp.replace(dest)
            return {"job": job["id"], "kind": kind, "status": "downloaded"}
        except Exception as e:  # GEE restricted mode: back off and retry
            last = str(e)[:300]
            time.sleep([5, 15, 45, 90, 180][attempt])
    return {"job": job["id"], "kind": kind, "status": "failed", "error": last}


def assemble(loc: pd.DataFrame, cache: Path) -> pd.DataFrame:
    out = loc.set_index("sample_id")[["longitude", "latitude"]].copy()
    for kind in ("alphaearth", "environment", "era5", "built", "worldpop"):
        frames = []
        for p in sorted(cache.glob(f"*_{kind}.json.gz")):
            with gzip.open(p, "rt", encoding="utf-8") as f:
                frames.append(pd.DataFrame(json.load(f)))
        t = pd.concat(frames, ignore_index=True).set_index("sample_id")
        if kind == "built":
            out["built_fraction"] = t["built_fraction"]
            out["built_coverage"] = t["built_coverage"]
        elif kind == "worldpop":
            avail = t["country_support_pixels"].fillna(0) > 0
            out["worldpop_2024_density"] = (t["population_count"].fillna(0) / t["buffer_area_m2"] * 1e6).where(avail)
            out["worldpop_country_available"] = avail
        else:
            drop = [c for c in t.columns if c in ("built_fraction", "built_coverage", "worldpop_2024_density")]
            out = out.join(t.drop(columns=drop), how="left")
    for n in range(1, 20):
        with rasterio.open(ROOT / "data/raw/worldclim_2_1_2_5m" / f"wc2.1_2.5m_bio_{n}.tif") as ds:
            vals = [v[0] for v in ds.sample(zip(out.longitude, out.latitude), masked=True)]
        out[f"bio19_{n}"] = [np.nan if np.ma.is_masked(v) else float(v) for v in vals]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--check-existing", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.out / "cache"
    cache.mkdir(exist_ok=True)
    loc = pd.read_csv(a.locations)[["sample_id", "longitude", "latitude"]]
    if a.check_existing:
        ref = pd.read_pickle(g.O / "point_features.pkl")
        neon = pd.read_csv(ROOT / "webapp/runtime/candidates/admission_2026_11_12/heldout_neon2026/annual_sites.csv")
        chk = neon[neon.sample_id.isin(ref.index)].sample(a.check_existing, random_state=0)[["sample_id", "longitude", "latitude"]]
        chk["sample_id"] = "chk__" + chk["sample_id"]
        loc = pd.concat([loc, chk], ignore_index=True)
    g.init()
    jobs = jobs_for(loc)
    tasks = [(j, k) for j in jobs for k in ("alphaearth", "environment", "era5", "built", "worldpop")]
    t0, states = time.time(), []
    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        for s in pool.map(lambda jk: fetch(jk[0], jk[1], cache), tasks):
            states.append(s)
            if len(states) % 10 == 0 or s["status"] == "failed":
                print(len(states), "/", len(tasks), s, f"{time.time() - t0:.0f}s", flush=True)
    (a.out / "acquisition.json").write_text(json.dumps(states, indent=1))
    if any(s["status"] == "failed" for s in states):
        print("Some requests failed; rerun to resume from cache.")
        return
    feats = assemble(loc, cache)
    if a.check_existing:
        ref = pd.read_pickle(g.O / "point_features.pkl")
        chk_ids = [i for i in feats.index if i.startswith("chk__")]
        cols = g.BIO + g.ENV + g.AE
        diff = (feats.loc[chk_ids, cols].set_axis([i[5:] for i in chk_ids]) - ref.loc[[i[5:] for i in chk_ids], cols]).abs()
        report = diff.max().sort_values(ascending=False)
        report.to_csv(a.out / "recipe_check_max_abs_diff.csv")
        print("Recipe reproduction check, largest max-abs differences:\n", report.head(10))
        feats = feats.drop(index=chk_ids)
    feats.to_pickle(a.out / "features.pkl")
    print("features", feats.shape, "complete rows", int(feats[g.BIO + g.ENV + g.AE].notna().all(axis=1).sum()))


if __name__ == "__main__":
    main()
