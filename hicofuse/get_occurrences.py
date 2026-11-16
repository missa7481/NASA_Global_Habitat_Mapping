#!/usr/bin/env python3
"""Download iNaturalist research-grade Culicidae occurrences (GBIF, 2022-2025).

Public GBIF occurrence-search API; no credentials or user details are sent.
Saved to data/raw/inat_culicidae_2022_2025/occurrences.csv with a manifest.
Records are later screened (coordinate uncertainty <= 2,000 m) and gridded.
NOTE: the search API is not a citable frozen download; a formal GBIF download
DOI must be requested before publication.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/raw/inat_culicidae_2022_2025"
API = "https://api.gbif.org/v1/occurrence/search"
BASE = {"datasetKey": "50c9509d-22c7-4a22-a47d-8c48425ef4a7", "taxonKey": 3346, "hasCoordinate": "true",
        "hasGeospatialIssue": "false", "limit": 300}
FIELDS = ["key", "species", "genus", "scientificName", "taxonRank", "decimalLatitude", "decimalLongitude",
          "coordinateUncertaintyInMeters", "year", "month", "countryCode", "eventDate"]


def get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(url)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in (2022, 2023, 2024, 2025):
        offset = 0
        while True:
            d = get({**BASE, "year": year, "offset": offset})
            rows += [{k: r.get(k) for k in FIELDS} for r in d["results"]]
            offset += len(d["results"])
            if d["endOfRecords"] or not d["results"]:
                break
        print(year, "cumulative", len(rows), flush=True)
    df = pd.DataFrame(rows).drop_duplicates("key")
    df.to_csv(OUT / "occurrences.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "retrieved_utc": datetime.now(timezone.utc).isoformat(), "api": API, "query": BASE,
        "years": [2022, 2025], "records": int(len(df)),
        "license_note": "iNaturalist records on GBIF carry CC0/CC-BY/CC-BY-NC per record; check before redistribution.",
        "citation_note": "Search-API snapshot; request a GBIF download DOI before publication.",
    }, indent=1))
    print("records", len(df))


if __name__ == "__main__":
    main()
