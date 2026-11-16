#!/usr/bin/env python3
"""Prepare AIMSurv 2020 (GBIF 10.15470/vs3677, CC0) trap-level Ae. albopictus labels.

AIMSurv is a harmonised pan-European ovitrap surveillance (46 teams, 24
countries) with explicit non-detections, and, unlike NEON, it samples the
urban-to-rural human-footprint gradient where reporting bias acts.  It is
prepared here as a *future held-out survey endpoint*; no model has seen it.

Event label rules (mirror the NEON 2025/2026 policy):
  positive   : an Ae. albopictus 'Present' row in the event
  negative   : the event has only 'Culicidae Absent' rows (zero eggs), or only
               identified non-albopictus species
  ambiguous  : unidentified 'Culicidae'/'Aedes' 'Present' and no albopictus -> excluded
Trap-level annual label: detected if >=1 positive event; non-detected only with
>=4 negative events spanning >=30 days and no positives; otherwise excluded.
Survey year 2020 vs 2024 covariates: Ae. albopictus has expanded since 2020, so
some 2020 non-detections may be 2024 presences (conservative for ranking).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "aimsurv_2020"
OUT = ROOT / "data" / "derived" / "aimsurv_2020"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    occ_path = RAW / "dwca" / "occurrence.txt"
    oc = pd.read_csv(occ_path, sep="\t", low_memory=False)
    oc["is_alb"] = oc["scientificName"].str.contains("albopictus", na=False)
    oc["present"] = oc["occurrenceStatus"].str.lower().eq("present")
    oc["unidentified"] = oc["taxonRank"].str.lower().isin(["family", "genus"]) | oc["scientificName"].isin(["Culicidae", "Aedes"])
    start = pd.to_datetime(oc["eventDate"].str.split("/").str[0], errors="coerce")
    oc["start"] = start
    for col in ("decimalLatitude", "decimalLongitude"):
        oc[col] = pd.to_numeric(oc[col], errors="coerce")
    oc = oc.dropna(subset=["decimalLatitude", "decimalLongitude"])

    ev = oc.groupby("eventID").agg(
        trap=("parentEventID", "first"), site=("locationID", "first"), country=("countryCode", "first"),
        lat=("decimalLatitude", "first"), lon=("decimalLongitude", "first"), habitat=("habitat", "first"),
        start=("start", "first"), effort=("samplingEffort", "first"),
        alb=("is_alb", lambda s: bool((s & oc.loc[s.index, "present"]).any())),
        unid_present=("unidentified", lambda s: bool((s & oc.loc[s.index, "present"]).any())),
    )
    ev["label"] = np.where(ev["alb"], "positive", np.where(ev["unid_present"], "ambiguous", "negative"))
    ev["effort_days"] = pd.to_numeric(ev["effort"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    ev.to_csv(OUT / "events.csv")

    usable = ev[ev["label"] != "ambiguous"]
    tr = usable.groupby("trap").agg(
        site=("site", "first"), country=("country", "first"), latitude=("lat", "median"), longitude=("lon", "median"),
        lat_sd=("lat", "std"), habitat=("habitat", "first"), n_events=("label", "size"),
        k_events=("label", lambda s: int((s == "positive").sum())),
        span_days=("start", lambda s: (s.max() - s.min()).days), effort_days=("effort_days", "sum"),
    )
    tr["target"] = (tr["k_events"] > 0).astype(int)
    eligible = tr["target"].eq(1) | (tr["n_events"].ge(4) & tr["span_days"].ge(30))
    traps = tr[eligible].copy()
    traps["sample_id"] = [
        "aim_" + hashlib.sha1(f"{a:.6f},{b:.6f}".encode()).hexdigest()[:16] for a, b in zip(traps.longitude, traps.latitude)
    ]
    traps = traps.reset_index().drop_duplicates("sample_id")
    traps.to_csv(OUT / "trap_sites.csv", index=False)
    traps[["sample_id", "longitude", "latitude"]].to_csv(OUT / "extraction_locations.csv", index=False)

    manifest = {
        "source": "AIMSurv, GBIF dataset 03269e13-84ae-430f-990e-f11069413e36, DOI 10.15470/vs3677, CC0",
        "occurrence_sha256": sha(occ_path),
        "events": int(len(ev)), "event_labels": ev["label"].value_counts().to_dict(),
        "traps_total": int(len(tr)), "traps_eligible": int(len(traps)),
        "traps_detected": int(traps["target"].sum()), "site_clusters": int(traps["site"].nunique()),
        "countries": int(traps["country"].nunique()),
        "habitat": traps["habitat"].value_counts().to_dict(),
        "by_country": traps.groupby("country")["target"].agg(["size", "sum"]).astype(int).to_dict("index"),
        "status": "labels prepared; covariates not yet extracted; never used to fit or select models",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "by_country"}, indent=1, default=str))


if __name__ == "__main__":
    main()
