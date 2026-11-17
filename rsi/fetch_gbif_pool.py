#!/usr/bin/env python3
"""Exogenous species-pool layer for all classes: GBIF Culicidae record counts by genus per 5-degree tile.

Records 1900-2021 only (the iNaturalist training labels are 2022-2025; pre-2022 iNaturalist is 0.6% of these
records). Aggregate counts only: no record coordinates are stored. Cached per tile, so re-runs resume.

  python rsi/fetch_gbif_pool.py   -> artifacts/rsi/gbif_pool/tiles.json
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicofuse.train import ART as HART  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts/rsi/gbif_pool"
OUT.mkdir(parents=True, exist_ok=True)
GENUS = {"Aedes": 7924646, "Culex": 1497010, "Anopheles": 1650098}
URL = ("https://api.gbif.org/v1/occurrence/search?taxonKey=3346&limit=0&facet=genusKey&facetLimit=200"
       "&year=1900,2021&hasGeospatialIssue=false&decimalLatitude={a},{b}&decimalLongitude={c},{d}")


def land_tiles():
    d = np.load(HART / "dataset.npz")
    rc = d["rc"][np.flatnonzero(d["complete"])]
    lat = 85 - (rc[:, 0] + 0.5) * 0.05
    lon = -180 + (rc[:, 1] + 0.5) * 0.05
    t = np.unique(np.c_[np.floor(lat / 5), np.floor(lon / 5)].astype(int), axis=0)
    return [(int(i), int(j)) for i, j in t]


def get(url, tries=8):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "habitat-research-client"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            ra = e.headers.get("Retry-After")
            time.sleep(float(ra) if ra and ra.isdigit() else min(120, 5 * 2 ** k))
            err = e
        except Exception as e:  # noqa: BLE001
            time.sleep(min(120, 5 * 2 ** k))
            err = e
    raise err


def main():
    path = OUT / "tiles.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    tiles = land_tiles()
    print(len(tiles), "tiles;", len(cache), "cached", flush=True)
    from concurrent.futures import ThreadPoolExecutor

    def one(t):
        i, j = t
        r = get(URL.format(a=i * 5, b=i * 5 + 5, c=j * 5, d=j * 5 + 5))
        fac = {int(c["name"]): c["count"] for c in r["facets"][0]["counts"]} if r["facets"] else {}
        return f"{i},{j}", {"total": r["count"], **{g: fac.get(k, 0) for g, k in GENUS.items()}}

    todo = [t for t in tiles if f"{t[0]},{t[1]}" not in cache]
    with ThreadPoolExecutor(1) as ex:
        for n, (key, v) in enumerate(ex.map(one, todo)):
            cache[key] = v
            path.write_text(json.dumps(cache))
            time.sleep(0.3)
            if n % 50 == 0:
                print(n, "/", len(todo), key, v, flush=True)
    path.write_text(json.dumps(cache))
    tot = {g: sum(v[g] for v in cache.values()) for g in ("total", *GENUS)}
    print("done", len(cache), tot, flush=True)


if __name__ == "__main__":
    main()
