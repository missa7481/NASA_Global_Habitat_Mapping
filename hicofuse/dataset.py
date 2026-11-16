#!/usr/bin/env python3
"""Assemble the grid-native multi-species training set for HiCo-Fuse.

Unit of analysis = 0.05-degree map cell (the same support as the global map).
  * Reports: iNaturalist research-grade Culicidae 2022-2025, coordinate uncertainty <= 2,000 m.
  * Classes: Ae. albopictus, Ae. aegypti, Culex spp., Anopheles spp., other Culicidae (reference).
  * Coarse level: 0.5-degree parent = 10 x 10 children. Every complete child cell of every parent
    that holds a report is kept, because the coherent model normalises over all children.
  * Spatial CV: 10-degree blocks dealt to 5 folds (seeded).
  * External: Kraemer et al. (2015) global compendium cells for Ae. aegypti / Ae. albopictus
    (literature records, mostly 1960-2014; never used for fitting).
Output: artifacts/hicofuse/dataset.npz + dataset_meta.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parents[1] / "artifacts/hicofuse"
CLASSES = ["Ae_albopictus", "Ae_aegypti", "Culex", "Anopheles", "other"]
W = 7200
PARENT = 10      # 10 x 0.05 = 0.5 degree
BLOCK = 200      # 200 x 0.05 = 10 degree CV blocks

MODALITIES = {
    "Climate": [f"bio19_{i}" for i in range(1, 20)] + ["air_temperature_c", "dewpoint_c", "soil_water_m3m3",
                                                       "era5_precip_mm", "temp_month_sd", "precip_month_cv", "rh_month_proxy"],
    "Landscape": ["elevation_m", "slope_deg", "NDVI", "EVI", "forest", "cropland", "urban", "wetland",
                  "lst_day_c", "lst_night_c", "surface_water_fraction_static2021"],
    "Human": ["worldpop_2024_density", "built_fraction", "night_lights", "accessibility_2015_minutes"],
    "AlphaEarth": [f"A{i:02d}" for i in range(64)],
}
LOG1P = ["worldpop_2024_density", "night_lights", "accessibility_2015_minutes", "era5_precip_mm",
         "bio19_12", "bio19_13", "bio19_14", "bio19_16", "bio19_17", "bio19_18", "bio19_19"]


def cell_of(lat, lon):
    r = np.floor((85 - np.asarray(lat)) / 0.05).astype(np.int64)
    c = np.floor((np.asarray(lon) + 180) / 0.05).astype(np.int64)
    return r, c


def classify(df: pd.DataFrame) -> np.ndarray:
    out = np.full(len(df), 4)
    out[df["genus"].eq("Culex").to_numpy()] = 2
    out[df["genus"].eq("Anopheles").to_numpy()] = 3
    out[df["species"].eq("Aedes aegypti").to_numpy()] = 1
    out[df["species"].eq("Aedes albopictus").to_numpy()] = 0
    return out


def main() -> None:
    meta = json.loads((ART / "store/meta.json").read_text())
    feats = meta["features"]
    X = np.load(ART / "store/X.npy", mmap_mode="r")
    rc = np.load(ART / "store/rc.npy")
    key = rc[:, 0].astype(np.int64) * W + rc[:, 1]
    complete = np.isfinite(np.asarray(X)).all(1)
    print("cells", len(key), "complete", complete.sum(), flush=True)
    order = np.argsort(key)
    skey = key[order]

    def lookup(r, c):
        k = r * W + c
        pos = np.searchsorted(skey, k)
        pos = np.clip(pos, 0, len(skey) - 1)
        hit = skey[pos] == k
        idx = np.where(hit, order[pos], -1)
        return idx

    occ = pd.read_csv(ROOT / "data/raw/inat_culicidae_2022_2025/occurrences.csv")
    occ = occ[occ["coordinateUncertaintyInMeters"].le(2000)].copy()
    occ["cls"] = classify(occ)
    r, c = cell_of(occ["decimalLatitude"], occ["decimalLongitude"])
    occ["idx"] = lookup(r, c)
    n_raw = len(occ)
    occ = occ[(occ["idx"] >= 0)]
    occ = occ[complete[occ["idx"].to_numpy()]]
    counts = pd.crosstab(occ["idx"], occ["cls"]).reindex(columns=range(5), fill_value=0)
    obs_idx = counts.index.to_numpy()
    Y = counts.to_numpy().astype(np.int32)

    # Parents and all their complete children
    pr = rc[:, 0] // PARENT
    pc = rc[:, 1] // PARENT
    pkey = pr.astype(np.int64) * (W // PARENT) + pc
    obs_parents = np.unique(pkey[obs_idx])
    child_mask = np.isin(pkey, obs_parents) & complete
    child_idx = np.flatnonzero(child_mask)
    # block folds
    bkey = (rc[:, 0] // BLOCK).astype(np.int64) * 100 + rc[:, 1] // BLOCK
    blocks = np.unique(bkey[obs_idx])
    rng = np.random.default_rng(20261114)
    fold_of_block = dict(zip(blocks, rng.permutation(len(blocks)) % 5))
    fold = np.array([fold_of_block[b] for b in bkey[obs_idx]])

    # Kraemer external cells
    kr = {}
    base = ROOT / "data/raw/mosquito_observations/kraemer_registered_archive"
    for sp in ("aegypti", "albopictus"):
        k = pd.read_csv(base / f"{sp}_rescued_occurrence.tsv", sep="\t", usecols=["decimalLatitude", "decimalLongitude", "year"], low_memory=False)
        k = k.dropna(subset=["decimalLatitude", "decimalLongitude"])
        ki = lookup(*cell_of(k["decimalLatitude"], k["decimalLongitude"]))
        ki = np.unique(ki[(ki >= 0)])
        kr[sp] = ki[complete[ki]]

    np.savez_compressed(
        ART / "dataset.npz", obs_idx=obs_idx, Y=Y, fold=fold, child_idx=child_idx,
        pkey=pkey.astype(np.int64), complete=complete, rc=rc,
        kraemer_aegypti=kr["aegypti"], kraemer_albopictus=kr["albopictus"],
    )
    col = {m: [feats.index(f) for f in fs] for m, fs in MODALITIES.items()}
    info = {
        "classes": CLASSES, "modalities": MODALITIES, "modality_cols": col,
        "log1p_cols": [feats.index(f) for f in LOG1P], "features": feats,
        "records_screened": int(n_raw), "records_on_complete_cells": int(len(occ)),
        "report_cells": int(len(obs_idx)), "class_records": dict(zip(CLASSES, Y.sum(0).tolist())),
        "class_cells": dict(zip(CLASSES, (Y > 0).sum(0).tolist())),
        "parents_with_reports": int(len(obs_parents)), "child_cells_in_those_parents": int(len(child_idx)),
        "cv_blocks": int(len(blocks)), "fold_report_cells": np.bincount(fold).tolist(),
        "kraemer_cells": {k: int(len(v)) for k, v in kr.items()},
    }
    (ART / "dataset_meta.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k not in ("modalities", "modality_cols", "features", "log1p_cols")}, indent=1))


if __name__ == "__main__":
    main()
