#!/usr/bin/env python3
"""Materialize a reproducible site-feature and representation-comparison bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPRESENTATION_ORDER = (
    "BioClim19",
    "Environment41",
    "AlphaEarth64",
    "Environment_AlphaEarth105",
)
METADATA_FIELDS = (
    "sample_id",
    "longitude",
    "latitude",
    "countryCode",
    "coordinateUncertaintyInMeters",
    "source",
    "datasetKey",
    "target",
    "year",
    "sample_role",
    "label_semantics",
    "provider_group",
    "usable_events",
    "trap_days",
    "detected_events",
    "egg_count",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value: str, field: str, sample_id: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{sample_id}: {field} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{sample_id}: {field} is not finite: {value!r}")
    return number


def primary_site_rows(
    metadata: list[dict[str, str]],
    features: list[dict[str, str]],
    representations: dict[str, list[str]],
    expected_primary_count: int = 1601,
) -> list[dict[str, str]]:
    feature_columns: list[str] = []
    for name in REPRESENTATION_ORDER:
        if name not in representations:
            raise ValueError(f"Protocol is missing representation {name!r}")
        if len(representations[name]) != len(set(representations[name])):
            raise ValueError(f"Representation {name!r} contains duplicate columns")
        feature_columns.extend(representations[name])

    feature_columns = list(dict.fromkeys(feature_columns))

    metadata_by_id: dict[str, dict[str, str]] = {}
    for row in metadata:
        sample_id = row.get("sample_id", "")
        if not sample_id:
            raise ValueError("Metadata contains an empty sample_id")
        if sample_id in metadata_by_id:
            raise ValueError(f"Duplicate metadata sample_id: {sample_id}")
        metadata_by_id[sample_id] = row

    feature_by_id: dict[str, dict[str, str]] = {}
    for row in features:
        sample_id = row.get("sample_id", "")
        if not sample_id:
            raise ValueError("Feature table contains an empty sample_id")
        if sample_id in feature_by_id:
            raise ValueError(f"Duplicate feature sample_id: {sample_id}")
        feature_by_id[sample_id] = row

    primary = [
        row for row in metadata
        if row.get("primary", "").strip().lower() == "true"
    ]
    if len(primary) != expected_primary_count:
        raise ValueError(
            f"Expected {expected_primary_count:,} primary locations, found {len(primary):,}"
        )

    result: list[dict[str, str]] = []
    for record in primary:
        sample_id = record["sample_id"]
        feature_row = feature_by_id.get(sample_id)
        if feature_row is None:
            raise ValueError(f"Missing features for primary sample {sample_id}")
        if record.get("target") not in {"0", "1"}:
            raise ValueError(f"{sample_id}: target must be 0 or 1")
        if record.get("year") != "2024":
            raise ValueError(f"{sample_id}: expected reference year 2024")

        row = {field: record.get(field, "") for field in METADATA_FIELDS}
        longitude = finite_number(row["longitude"], "longitude", sample_id)
        latitude = finite_number(row["latitude"], "latitude", sample_id)
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise ValueError(f"{sample_id}: coordinates are outside valid ranges")

        for column in feature_columns:
            if column not in feature_row:
                raise ValueError(f"Feature table is missing required column {column!r}")
            finite_number(feature_row[column], column, sample_id)
            row[column] = feature_row[column]
        result.append(row)
    return result


def coverage_features(rows: list[dict[str, str]], cell_degrees: int = 5) -> dict[str, Any]:
    cells: dict[tuple[int, int], dict[str, Any]] = defaultdict(
        lambda: {"n_sites": 0, "n_presence": 0, "n_background": 0, "sources": Counter()}
    )
    for row in rows:
        longitude = finite_number(row["longitude"], "longitude", row["sample_id"])
        latitude = finite_number(row["latitude"], "latitude", row["sample_id"])
        x = min(360 // cell_degrees - 1, max(0, math.floor((longitude + 180) / cell_degrees)))
        y = min(180 // cell_degrees - 1, max(0, math.floor((latitude + 90) / cell_degrees)))
        cell = cells[(x, y)]
        cell["n_sites"] += 1
        cell["n_presence" if row["target"] == "1" else "n_background"] += 1
        cell["sources"][row.get("source") or "unknown"] += 1

    features = []
    for (x, y), counts in sorted(cells.items()):
        west = -180 + x * cell_degrees
        south = -90 + y * cell_degrees
        east = west + cell_degrees
        north = south + cell_degrees
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [west, south],
                        [east, south],
                        [east, north],
                        [west, north],
                        [west, south],
                    ]],
                },
                "properties": {
                    "n_sites": counts["n_sites"],
                    "n_presence": counts["n_presence"],
                    "n_background": counts["n_background"],
                    "sources": dict(sorted(counts["sources"].items())),
                    "cell_degrees": cell_degrees,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def source_coverage_features(
    training_rows: list[dict[str, str]],
    mosquito_alert_rows: list[dict[str, str]],
    globe_rows: list[dict[str, str]],
    neon_rows: list[dict[str, str]],
    cell_degrees: int = 5,
) -> dict[str, Any]:
    categories = (
        "training_sites",
        "training_presence",
        "training_background",
        "mosquito_alert_candidates",
        "globe_habitat_observations",
        "globe_species_claims",
        "globe_verified_species_claims",
        "neon_validation_sites",
        "neon_detections",
    )
    cells: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)

    def add(
        row: dict[str, str],
        category: str,
        *,
        id_field: str = "sample_id",
        include: bool = True,
    ) -> None:
        if not include:
            return
        record_id = row.get(id_field) or "candidate record"
        longitude = finite_number(row.get("longitude", ""), "longitude", record_id)
        latitude = finite_number(row.get("latitude", ""), "latitude", record_id)
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise ValueError(f"{record_id}: coordinates are outside valid ranges")
        x = min(
            360 // cell_degrees - 1,
            max(0, math.floor((longitude + 180) / cell_degrees)),
        )
        y = min(
            180 // cell_degrees - 1,
            max(0, math.floor((latitude + 90) / cell_degrees)),
        )
        cells[(x, y)][category] += 1

    for row in training_rows:
        add(row, "training_sites")
        add(row, "training_presence" if row["target"] == "1" else "training_background")

    for row in mosquito_alert_rows:
        add(
            row,
            "mosquito_alert_candidates",
            include=row.get("candidate_eligible", "").lower() == "true",
        )

    for row in globe_rows:
        if row.get("coordinates_valid", "").lower() != "true":
            continue
        add(row, "globe_habitat_observations", id_field="observation_id")
        if row.get("species_claim", "").lower() == "true":
            add(row, "globe_species_claims", id_field="observation_id")
        if row.get("expert_verified", "").lower() == "true":
            add(row, "globe_verified_species_claims", id_field="observation_id")

    for row in neon_rows:
        if row.get("independent_gt10km", "").lower() != "true":
            continue
        add(row, "neon_validation_sites")
        if row.get("target") == "1":
            add(row, "neon_detections")

    features = []
    for (x, y), counts in sorted(cells.items()):
        west = -180 + x * cell_degrees
        south = -90 + y * cell_degrees
        east = west + cell_degrees
        north = south + cell_degrees
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [west, south],
                        [east, south],
                        [east, north],
                        [west, north],
                        [west, south],
                    ]],
                },
                "properties": {
                    **{category: counts[category] for category in categories},
                    "cell_degrees": cell_degrees,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def build_source_inventory(
    training_rows: list[dict[str, str]],
    mosquito_alert_rows: list[dict[str, str]],
    globe_rows: list[dict[str, str]],
    neon_rows: list[dict[str, str]],
    gbif_admission: list[dict[str, Any]],
    mosquito_alert_admission: dict[str, Any],
) -> dict[str, Any]:
    valid_globe = [
        row for row in globe_rows
        if row.get("coordinates_valid", "").lower() == "true"
    ]
    independent_neon = [
        row for row in neon_rows
        if row.get("independent_gt10km", "").lower() == "true"
    ]
    mosquito_years = {
        str(row.get("year")): row
        for row in mosquito_alert_admission.get("years", [])
    }
    gbif_rows = {
        (row.get("species"), str(row.get("year"))): row
        for row in gbif_admission
    }
    primary_countries = {
        row.get("countryCode") for row in training_rows if row.get("countryCode")
    }
    mosquito_countries = {
        row.get("countryCode") for row in mosquito_alert_rows if row.get("countryCode")
    }
    globe_countries = {
        row.get("country") for row in valid_globe if row.get("country")
    }

    return {
        "species": "Aedes albopictus",
        "reference_year": 2024,
        "coverage_cell_degrees": 5,
        "sources": [
            {
                "id": "training_sites",
                "name": "Frozen development cohort",
                "role": "Primary 2024 presence-background training and spatial development",
                "locations": len(training_rows),
                "presence": sum(row["target"] == "1" for row in training_rows),
                "target_group_background": sum(
                    row["target"] == "0" for row in training_rows
                ),
                "country_count": len(primary_countries),
                "map_layer": "training_sites",
                "source": "https://www.gbif.org/",
                "limitation": (
                    "Target-group background locations are not confirmed absences; "
                    "records reflect uneven observer effort."
                ),
            },
            {
                "id": "mosquito_alert_candidates",
                "name": "Mosquito Alert 2024 candidate subset",
                "role": "Exploratory presence-only candidate; not used for training",
                "locations": sum(
                    row.get("candidate_eligible", "").lower() == "true"
                    for row in mosquito_alert_rows
                ),
                "country_count": len(mosquito_countries),
                "high_confidence_release_records": mosquito_years.get("2024", {}).get(
                    "high_confidence", 0
                ),
                "new_ids_vs_older_archive": mosquito_years.get("2024", {}).get(
                    "unseen_old2024", 0
                ),
                "release_years": [
                    {
                        "year": int(year),
                        "high_confidence_records": value.get("high_confidence", 0),
                        "unseen_older_archive": value.get("unseen_old2024", 0),
                    }
                    for year, value in sorted(mosquito_years.items())
                ],
                "map_layer": "mosquito_alert_candidates",
                "source": "https://doi.org/10.5281/zenodo.18751762",
                "limitation": (
                    "The selected 2024 subset lacks numeric location uncertainty and "
                    "requires cross-provider deduplication; not independent validation."
                ),
            },
            {
                "id": "globe_habitat_observations",
                "name": "GLOBE Observer mosquito-habitat reports",
                "role": "Auxiliary habitat observations; not species-presence labels",
                "observations": len(globe_rows),
                "coordinate_valid_observations": len(valid_globe),
                "country_count": len(globe_countries),
                "species_claims": sum(
                    row.get("species_claim", "").lower() == "true"
                    for row in valid_globe
                ),
                "expert_verified_species_claims": sum(
                    row.get("expert_verified", "").lower() == "true"
                    for row in valid_globe
                ),
                "map_layer": "globe_habitat_observations",
                "source": "https://observer.globe.gov/get-data",
                "limitation": (
                    "Habitat observations do not establish Aedes albopictus presence; "
                    "species claims in this archive are not expert-verified."
                ),
            },
            {
                "id": "neon_validation_sites",
                "name": "NEON 2026 external survey diagnostic",
                "role": "Temporal survey diagnostic; not used for training or model selection",
                "locations": len(independent_neon),
                "detections": sum(row.get("target") == "1" for row in independent_neon),
                "map_layer": "neon_validation_sites",
                "source": "https://data.neonscience.org/data-products/DP1.10043.001",
                "limitation": (
                    "United States surveillance with method- and year-specific labels; "
                    "previously examined and not a blind global validation."
                ),
            },
            {
                "id": "gbif_2024_screen",
                "name": "GBIF 2024 occurrence screen",
                "role": "Source audit; records overlap the existing archive",
                "records": gbif_rows.get(("albopictus", "2024"), {}).get("count", 0),
                "coordinate_precision_le_500m": gbif_rows.get(
                    ("albopictus", "2024"), {}
                ).get("precision_le500", 0),
                "unseen_local_precision_screen_pass": gbif_rows.get(
                    ("albopictus", "2024"), {}
                ).get("unseen_local_screen_pass", 0),
                "map_layer": None,
                "source": "https://www.gbif.org/",
                "limitation": (
                    "The precision-screened 2024 records were already present in the "
                    "local archive; do not count them as new independent sites."
                ),
            },
            {
                "id": "gbif_2026_reserved",
                "name": "GBIF 2026 occurrence screen",
                "role": "Reserved presence-only candidate; not used for training",
                "records": gbif_rows.get(("albopictus", "2026"), {}).get("count", 0),
                "coordinate_precision_le_500m": gbif_rows.get(
                    ("albopictus", "2026"), {}
                ).get("precision_le500", 0),
                "map_layer": None,
                "source": "https://www.gbif.org/",
                "limitation": (
                    "No DOI-frozen download snapshot; presence-only records are not "
                    "surveyed absences or independent validation."
                ),
            },
        ],
        "abundance": {
            "status": "not mapped",
            "reason": (
                "Available trap/count sources differ in effort, season, life stage, "
                "and spatial support; no harmonized globally dense target is available."
            ),
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def build_bundle(source_root: Path, output_dir: Path) -> dict[str, Any]:
    base = source_root / "data" / "derived" / "global_mapping_2024"
    metadata_path = base / "models" / "development_metadata.csv"
    feature_path = base / "point_features.csv"
    protocol_path = base / "models" / "protocol.json"
    summary_path = base / "models" / "summary.csv"
    fold_metrics_path = base / "models" / "fold_metrics.csv"
    candidate_root = (
        source_root / "webapp" / "runtime" / "candidates" / "admission_2026_11_12"
    )
    mosquito_alert_path = (
        source_root / "data" / "derived" / "habitat_extension_2026_11_04"
        / "mosquitoalert_sites.csv"
    )
    globe_path = candidate_root / "globe_observations.csv"
    neon_path = candidate_root / "global_external2026" / "external_evaluated_sites.csv"
    gbif_admission_path = candidate_root / "gbif_admission.json"
    mosquito_alert_admission_path = candidate_root / "mosquitoalert_admission.json"
    inputs = (
        metadata_path,
        feature_path,
        protocol_path,
        summary_path,
        fold_metrics_path,
        mosquito_alert_path,
        globe_path,
        neon_path,
        gbif_admission_path,
        mosquito_alert_admission_path,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Required source file not found: {path}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    representations = protocol.get("representations", {})
    expected_dimensions = {
        "BioClim19": 19,
        "Environment41": 41,
        "AlphaEarth64": 64,
        "Environment_AlphaEarth105": 105,
    }
    if {
        name: len(representations.get(name, []))
        for name in expected_dimensions
    } != expected_dimensions:
        raise ValueError("Frozen protocol representation dimensions do not match the cohort contract")
    if set(representations["Environment_AlphaEarth105"]) != (
        set(representations["Environment41"]) | set(representations["AlphaEarth64"])
    ):
        raise ValueError("Fused representation must contain Environment41 + AlphaEarth64")

    rows = primary_site_rows(
        read_csv(metadata_path),
        read_csv(feature_path),
        representations,
    )
    mosquito_alert_rows = read_csv(mosquito_alert_path)
    globe_rows = read_csv(globe_path)
    neon_rows = read_csv(neon_path)
    gbif_admission = json.loads(gbif_admission_path.read_text(encoding="utf-8"))
    mosquito_alert_admission = json.loads(
        mosquito_alert_admission_path.read_text(encoding="utf-8")
    )
    labels = Counter(row["target"] for row in rows)
    if labels != Counter({"0": 862, "1": 739}):
        raise ValueError(f"Unexpected primary label counts: {dict(labels)}")

    feature_columns = list(
        dict.fromkeys(
            column
            for representation in REPRESENTATION_ORDER
            for column in representations[representation]
        )
    )
    if len(feature_columns) != 105:
        raise ValueError(f"Expected 105 unique predictor columns, found {len(feature_columns)}")
    dataset_path = output_dir / "dataset" / "site_features.csv"
    write_csv(dataset_path, rows, list(METADATA_FIELDS) + feature_columns)

    coverage_path = output_dir / "dataset" / "coverage_5deg.geojson"
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.write_text(
        json.dumps(coverage_features(rows), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    source_coverage_path = output_dir / "dataset" / "source_coverage_5deg.geojson"
    source_coverage_path.write_text(
        json.dumps(
            source_coverage_features(
                rows,
                mosquito_alert_rows,
                globe_rows,
                neon_rows,
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
    )
    source_inventory_path = output_dir / "dataset" / "source_inventory.json"
    source_inventory_path.write_text(
        json.dumps(
            build_source_inventory(
                rows,
                mosquito_alert_rows,
                globe_rows,
                neon_rows,
                gbif_admission,
                mosquito_alert_admission,
            ),
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    summary = [
        row for row in read_csv(summary_path)
        if row.get("endpoint") == "presence_background"
        and row.get("factor") in REPRESENTATION_ORDER
    ]
    summary.sort(
        key=lambda row: (
            REPRESENTATION_ORDER.index(row["factor"]),
            row["learner"],
        )
    )
    metric_path = output_dir / "results" / "representation_comparison.csv"
    write_csv(
        metric_path,
        summary,
        ["endpoint", "factor", "learner", "AP_mean", "AP_fold_SD", "AUC_mean", "folds"],
    )

    fold_rows = [
        row for row in read_csv(fold_metrics_path)
        if row.get("endpoint") == "presence_background"
        and row.get("factor") in REPRESENTATION_ORDER
        and row.get("learner") in {"GLM", "RF"}
    ]
    fold_index = {
        (row["factor"], row["learner"], row["context"]): float(row["AP"])
        for row in fold_rows
    }
    contexts = sorted({row["context"] for row in fold_rows})
    pair_rows: list[dict[str, Any]] = []
    for learner in ("GLM", "RF"):
        for left, right in (
            ("Environment41", "Environment_AlphaEarth105"),
            ("AlphaEarth64", "Environment_AlphaEarth105"),
        ):
            differences = [
                fold_index[(right, learner, context)] - fold_index[(left, learner, context)]
                for context in contexts
                if (right, learner, context) in fold_index
                and (left, learner, context) in fold_index
            ]
            if len(differences) != len(contexts):
                raise ValueError(
                    f"Incomplete paired folds for {learner}: {left} vs {right}"
                )
            mean_delta = sum(differences) / len(differences)
            variance = sum((value - mean_delta) ** 2 for value in differences) / max(
                1, len(differences) - 1
            )
            pair_rows.append(
                {
                    "endpoint": "presence_background",
                    "learner": learner,
                    "comparison": f"{right} minus {left}",
                    "mean_delta_AP": f"{mean_delta:.6f}",
                    "fold_delta_SD": f"{math.sqrt(variance):.6f}",
                    "n_spatial_folds": len(differences),
                    "interpretation": "Descriptive paired fold spread; not an independent-sample confidence interval",
                }
            )
    paired_path = output_dir / "results" / "paired_representation_deltas.csv"
    write_csv(
        paired_path,
        pair_rows,
        [
            "endpoint",
            "learner",
            "comparison",
            "mean_delta_AP",
            "fold_delta_SD",
            "n_spatial_folds",
            "interpretation",
        ],
    )

    file_records = {}
    for path in inputs:
        file_records[str(path.relative_to(source_root))] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    outputs = (
        dataset_path,
        coverage_path,
        source_coverage_path,
        source_inventory_path,
        metric_path,
        paired_path,
    )
    manifest = {
        "project": "Global Mosquito Habitat Foundations",
        "species": "Aedes albopictus",
        "reference_year": 2024,
        "cohort": {
            "sites": len(rows),
            "presence": labels["1"],
            "target_group_background": labels["0"],
            "feature_count": len(feature_columns),
            "representations": {
                name: len(representations[name]) for name in REPRESENTATION_ORDER
            },
        },
        "evaluation": {
            "endpoint": "spatial development presence-background",
            "metric_source": str(summary_path.relative_to(source_root)),
            "paired_folds": len(contexts),
            "note": "Frozen upstream results; this build does not refit models.",
        },
        "inputs": file_records,
        "outputs": {
            str(path.relative_to(output_dir)): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in outputs
        },
    }
    manifest_path = output_dir / "provenance.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def publish_aggregate_files(output_dir: Path, public_data_dir: Path) -> None:
    public_data_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("source_coverage_5deg.geojson", "source_inventory.json"):
        source = output_dir / "dataset" / filename
        if not source.is_file():
            raise FileNotFoundError(f"Aggregate publication input not found: {source}")
        shutil.copyfile(source, public_data_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Shared workspace root containing data\\derived\\global_mapping_2024",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
        help="Directory for generated local bundle files",
    )
    parser.add_argument(
        "--public-data-dir",
        type=Path,
        help="Optional website data directory; copies only aggregate coverage and inventory",
    )
    args = parser.parse_args()
    manifest = build_bundle(args.source_root.resolve(), args.output_dir.resolve())
    if args.public_data_dir:
        publish_aggregate_files(
            args.output_dir.resolve(),
            args.public_data_dir.resolve(),
        )
    print(
        f"Built {manifest['cohort']['sites']} sites, "
        f"{manifest['cohort']['feature_count']} features; "
        f"wrote {len(manifest['outputs'])} artifacts to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
