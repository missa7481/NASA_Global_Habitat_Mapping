#!/usr/bin/env python3
"""Compare shallow nonlinear representations on the frozen spatial folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODEL_CONFIG = {
    "activation": "relu",
    "alpha": 0.01,
    "batch_size": 128,
    "early_stopping": True,
    "learning_rate_init": 0.001,
    "max_iter": 300,
    "n_iter_no_change": 20,
    "solver": "adam",
    "validation_fraction": 0.15,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spatial_groups(rows: list[dict[str, str]], cell_degrees: int = 5) -> np.ndarray:
    groups = []
    for row in rows:
        longitude = float(row["longitude"])
        latitude = float(row["latitude"])
        if not math.isfinite(longitude) or not math.isfinite(latitude):
            raise ValueError(f"{row['sample_id']}: coordinates must be finite")
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise ValueError(f"{row['sample_id']}: coordinates are outside valid ranges")
        x = min(
            360 // cell_degrees - 1,
            max(0, math.floor((longitude + 180) / cell_degrees)),
        )
        y = min(
            180 // cell_degrees - 1,
            max(0, math.floor((latitude + 90) / cell_degrees)),
        )
        groups.append(f"{x}:{y}")
    return np.asarray(groups, dtype=str)


def make_model(hidden_layer_sizes: tuple[int, ...], random_state: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            **MODEL_CONFIG,
        ),
    )


def fit_model(model: Any, features: np.ndarray, labels: np.ndarray) -> Any:
    validation_size = math.ceil(
        len(labels) * MODEL_CONFIG["validation_fraction"]
    )
    train_size = max(1, len(labels) - validation_size)
    model.named_steps["mlpclassifier"].set_params(
        batch_size=min(MODEL_CONFIG["batch_size"], train_size)
    )
    model.fit(features, labels)
    return model


def positive_probability(model: Any, features: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("A fitted MLP did not contain positive training labels")
    return model.predict_proba(features)[:, classes.index(1)]


def spatial_oof_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    hidden_layer_sizes: tuple[int, ...],
    random_state: int,
    n_splits: int = 4,
) -> np.ndarray:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    oof = np.full(len(labels), np.nan, dtype=float)
    for split_index, (train_index, validation_index) in enumerate(
        splitter.split(features, labels, groups)
    ):
        if len(np.unique(labels[train_index])) != 2:
            raise ValueError(
                f"Inner spatial split {split_index} has only one training class"
            )
        model = make_model(
            hidden_layer_sizes,
            random_state + split_index + 1,
        )
        fit_model(model, features[train_index], labels[train_index])
        oof[validation_index] = positive_probability(
            model,
            features[validation_index],
        )
    if not np.isfinite(oof).all():
        raise ValueError("Spatial out-of-fold predictions are incomplete")
    return oof


def fold_metrics(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    context_id: str,
    n_train: int,
    n_test: int,
) -> list[dict[str, Any]]:
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{context_id}: outer test fold must contain both classes")
    return [
        {
            "context": context_id,
            "representation": name,
            "learner": "MLPClassifier",
            "n_train": n_train,
            "n_test": n_test,
            "positive": int(labels.sum()),
            "AP": float(average_precision_score(labels, score)),
            "AUC": float(roc_auc_score(labels, score)),
        }
        for name, score in probabilities.items()
    ]


def summarize_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_representation: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_representation.setdefault(row["representation"], []).append(row)
    summary = []
    for representation, folds in sorted(by_representation.items()):
        ap_values = [row["AP"] for row in folds]
        auc_values = [row["AUC"] for row in folds]
        summary.append(
            {
                "representation": representation,
                "learner": "MLPClassifier",
                "AP_mean": float(np.mean(ap_values)),
                "AP_fold_SD": float(np.std(ap_values, ddof=1)),
                "AUC_mean": float(np.mean(auc_values)),
                "folds": len(folds),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(
    source_root: Path,
    artifact_dir: Path,
    public_data_dir: Path | None = None,
) -> dict[str, Any]:
    base = source_root / "data" / "derived" / "global_mapping_2024"
    contexts_path = base / "models" / "contexts.json"
    protocol_path = base / "models" / "protocol.json"
    site_features_path = artifact_dir / "dataset" / "site_features.csv"
    frozen_summary_path = artifact_dir / "results" / "representation_comparison.csv"
    inputs = (contexts_path, protocol_path, site_features_path, frozen_summary_path)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Required benchmark input not found: {path}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    representations = protocol["representations"]
    environment_columns = representations["Environment41"]
    embedding_columns = representations["AlphaEarth64"]
    fused_columns = representations["Environment_AlphaEarth105"]
    if len(environment_columns) != 41 or len(embedding_columns) != 64:
        raise ValueError("Frozen experiment requires Environment41 and AlphaEarth64")
    if set(fused_columns) != set(environment_columns) | set(embedding_columns):
        raise ValueError("Frozen fused representation is not the expected 105 features")

    rows = read_csv(site_features_path)
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Primary site feature table contains duplicate sample IDs")
    row_index = {sample_id: index for index, sample_id in enumerate(ids)}
    labels = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    if len(rows) != 1601 or int(labels.sum()) != 739:
        raise ValueError("Primary cohort no longer matches the frozen 1,601 / 739 contract")

    def feature_matrix(columns: list[str]) -> np.ndarray:
        try:
            matrix = np.asarray(
                [[float(row[column]) for column in columns] for row in rows],
                dtype=np.float64,
            )
        except (KeyError, ValueError) as error:
            raise ValueError("Site feature table is missing valid numeric predictors") from error
        if matrix.shape != (len(rows), len(columns)) or not np.isfinite(matrix).all():
            raise ValueError("Feature matrix has an unexpected shape or non-finite values")
        return matrix

    environment = feature_matrix(environment_columns)
    embedding = feature_matrix(embedding_columns)
    fused = feature_matrix(fused_columns)
    groups = spatial_groups(rows)
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    if len(contexts) != 6:
        raise ValueError(f"Expected six frozen outer contexts, found {len(contexts)}")

    results: list[dict[str, Any]] = []
    for context_index, context in enumerate(contexts):
        train_ids = set(context["train_ids"]) & row_index.keys()
        test_ids = set(context["test_ids"]) & row_index.keys()
        if train_ids & test_ids:
            raise ValueError(f"{context['id']}: training and test IDs overlap")
        train_index = np.asarray([row_index[sample_id] for sample_id in ids if sample_id in train_ids])
        test_index = np.asarray([row_index[sample_id] for sample_id in ids if sample_id in test_ids])
        if len(train_index) == 0 or len(test_index) == 0:
            raise ValueError(f"{context['id']}: empty primary train or test set")
        y_train = labels[train_index]
        y_test = labels[test_index]
        if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
            raise ValueError(f"{context['id']}: train and test must each contain both classes")

        seed = 2026 + context_index * 100
        early_model = make_model((64, 32), seed)
        fit_model(early_model, fused[train_index], y_train)
        early_probability = positive_probability(early_model, fused[test_index])

        environment_model = make_model((32, 16), seed + 1)
        embedding_model = make_model((32, 16), seed + 2)
        environment_oof = spatial_oof_probabilities(
            environment[train_index],
            y_train,
            groups[train_index],
            (32, 16),
            seed + 10,
        )
        embedding_oof = spatial_oof_probabilities(
            embedding[train_index],
            y_train,
            groups[train_index],
            (32, 16),
            seed + 20,
        )
        fusion_features = np.column_stack((environment_oof, embedding_oof))
        meta_model = make_model((4,), seed + 30)
        fit_model(meta_model, fusion_features, y_train)

        fit_model(environment_model, environment[train_index], y_train)
        fit_model(embedding_model, embedding[train_index], y_train)
        environment_probability = positive_probability(
            environment_model,
            environment[test_index],
        )
        embedding_probability = positive_probability(
            embedding_model,
            embedding[test_index],
        )
        late_probability = positive_probability(
            meta_model,
            np.column_stack((environment_probability, embedding_probability)),
        )

        fold_results = fold_metrics(
            y_test,
            {
                "Environment41": environment_probability,
                "AlphaEarth64": embedding_probability,
                "Environment_AlphaEarth105": early_probability,
                "Nonlinear_late_fusion": late_probability,
            },
            context["id"],
            len(train_index),
            len(test_index),
        )
        results.extend(fold_results)

    summary = summarize_metrics(results)
    results_dir = artifact_dir / "results"
    fold_path = results_dir / "fusion_mlp_fold_metrics.csv"
    summary_path = results_dir / "fusion_mlp_summary.csv"
    write_csv(
        fold_path,
        results,
        ["context", "representation", "learner", "n_train", "n_test", "positive", "AP", "AUC"],
    )
    write_csv(
        summary_path,
        summary,
        ["representation", "learner", "AP_mean", "AP_fold_SD", "AUC_mean", "folds"],
    )

    frozen = [
        row for row in read_csv(frozen_summary_path)
        if row.get("endpoint") == "presence_background"
    ]
    display_names = {
        "Environment41": "Environment 41",
        "AlphaEarth64": "AlphaEarth 64",
        "Environment_AlphaEarth105": "Early fusion · Environment + AlphaEarth",
        "Nonlinear_late_fusion": "Nonlinear late fusion · stacked MLP",
    }
    public_summary = {
        "species": "Aedes albopictus",
        "reference_year": 2024,
        "training_sites": len(rows),
        "evaluation": {
            "endpoint": "presence-background ranking",
            "outer_spatial_folds": len(contexts),
            "preprocessing": "StandardScaler fit separately inside each training fold",
            "inner_fusion": (
                "Four-fold StratifiedGroupKFold using 5-degree spatial cells; "
                "nonlinear MLP meta-learner fit to out-of-fold modality scores"
            ),
            "notes": [
                "Background sites are not confirmed absences.",
                "Fold standard deviations are descriptive, not confidence intervals.",
                "Frozen MaxNet, GLM and RF results were not refit by this experiment.",
                "The MLP results are a point benchmark, not a released global prediction map.",
            ],
        },
        "training_protocol": {
            "feature_scaling": (
                "StandardScaler is fitted only on each model's training partition."
            ),
            "model_config": MODEL_CONFIG,
            "effective_batch_size": (
                "min(128, training partition size after the internal early-stopping split)"
            ),
            "architectures": {
                "Environment41": [32, 16],
                "AlphaEarth64": [32, 16],
                "early_fusion_Environment_AlphaEarth105": [64, 32],
                "late_fusion_experts": [32, 16],
                "late_fusion_meta_mlp": [4],
            },
            "random_seed_scheme": (
                "Base seed 2026 plus 100 per outer context, with fixed offsets per model."
            ),
        },
        "models": [
            {
                "representation": row["factor"],
                "display_name": (
                    f"{display_names.get(row['factor'], row['factor'])} · {row['learner']}"
                ),
                "learner": row["learner"],
                "AP_mean": float(row["AP_mean"]),
                "AP_fold_SD": float(row["AP_fold_SD"]),
                "AUC_mean": float(row["AUC_mean"]),
                "folds": int(row["folds"]),
                "source": "frozen upstream benchmark",
            }
            for row in frozen
        ] + [
            {
                **row,
                "display_name": (
                    f"{display_names.get(row['representation'], row['representation'])} · MLP"
                ),
                "source": "new fold-aligned MLP experiment",
            }
            for row in summary
        ],
    }
    public_summary_path = results_dir / "fusion_mlp_summary.json"
    public_summary_path.write_text(
        json.dumps(public_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    protocol_record = {
        "experiment": "Aedes albopictus shallow nonlinear representation comparison",
        "cohort_sites": len(rows),
        "cohort_presence": int(labels.sum()),
        "outer_contexts": [context["id"] for context in contexts],
        "inner_group_cell_degrees": 5,
        "inner_fusion_folds": 4,
        "random_seed_scheme": (
            "Base seed 2026 plus 100 per outer context, with fixed offsets per model."
        ),
        "model_config": MODEL_CONFIG,
        "batch_size_policy": "min(configured batch size, current training partition size)",
        "architectures": {
            "Environment41": [32, 16],
            "AlphaEarth64": [32, 16],
            "Environment_AlphaEarth105": [64, 32],
            "Nonlinear_late_fusion": {
                "modality_experts": [32, 16],
                "meta_learner": [4],
            },
        },
        "sklearn_version": sklearn.__version__,
        "inputs": {
            str(path.relative_to(source_root)): sha256(path)
            for path in inputs
        },
        "outputs": {
            str(path.relative_to(artifact_dir)): sha256(path)
            for path in (fold_path, summary_path, public_summary_path)
        },
    }
    protocol_out = results_dir / "fusion_mlp_protocol.json"
    protocol_out.write_text(
        json.dumps(protocol_record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if public_data_dir is not None:
        public_data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            public_summary_path,
            public_data_dir / "fusion_mlp_summary.json",
        )
    return public_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Shared workspace root containing data\\derived\\global_mapping_2024",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
        help="Directory containing the built foundation bundle",
    )
    parser.add_argument(
        "--public-data-dir",
        type=Path,
        help="Optional website data directory; copies aggregate metrics only",
    )
    args = parser.parse_args()
    result = run_benchmark(
        args.source_root.resolve(),
        args.artifact_dir.resolve(),
        args.public_data_dir.resolve() if args.public_data_dir else None,
    )
    print(
        f"Evaluated {len(result['models'])} frozen and MLP rows across "
        f"{result['evaluation']['outer_spatial_folds']} outer spatial folds"
    )


if __name__ == "__main__":
    main()
