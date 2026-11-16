"""Load the frozen Aedes albopictus cohort, Austrian survey and NEON 2026 sites.

Everything here is read-only with respect to the shared workspace.  The three
data blocks have different label semantics and are never pooled into a single
binary target:

* ``po``     : iNaturalist presences (y=1) vs. iNaturalist other-Culicidae
               target-group background (y=0).  Opportunistic; observer effort
               is retained in the labels.
* ``survey`` : Austrian 2024 ovitrap monitoring.  Per-site number of usable
               sampling events and number of events with species-confirmed
               eggs.  0 detections means non-detection over the visits, not a
               certain absence.
* ``neon``   : NEON 2026 provisional annual detection/non-detection.  External
               diagnostic only: never used to fit, tune or select models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[2]
BASE = WORKSPACE / "data" / "derived" / "global_mapping_2024"
EXTERNAL = (
    WORKSPACE / "webapp" / "runtime" / "candidates" / "admission_2026_11_12"
    / "global_external2026" / "external_evaluated_sites.csv"
)

# Covariates that describe where people are / where observers can reach.
# They are ALSO plausible habitat drivers for a peridomestic container breeder,
# which is exactly why the observation process is not identifiable from
# presence-only data alone.
EFFORT = ["worldpop_2024_density", "built_fraction", "night_lights", "accessibility_2015_minutes"]
LOG1P = {"worldpop_2024_density", "night_lights", "accessibility_2015_minutes", "era5_precip_mm"}


def _protocol() -> dict:
    return json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))


@dataclass
class Cohort:
    meta: pd.DataFrame            # indexed by sample_id; development sites
    X: pd.DataFrame               # raw 105 features, indexed by sample_id
    neon_meta: pd.DataFrame
    neon_X: pd.DataFrame
    env_cols: list[str]
    ae_cols: list[str]
    contexts: list[dict] = field(repr=False, default_factory=list)

    @property
    def env_eco_cols(self) -> list[str]:
        """Environment41 minus the four human-footprint / effort covariates."""
        return [c for c in self.env_cols if c not in EFFORT]


def transform(X: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, fold-independent monotone transform for skewed inputs."""
    out = X.copy()
    for c in out.columns:
        if c in LOG1P:
            out[c] = np.log1p(np.clip(out[c].astype(float), 0, None))
    return out


def load() -> Cohort:
    proto = _protocol()
    env_cols = list(proto["representations"]["Environment41"])
    ae_cols = list(proto["representations"]["AlphaEarth64"])
    cols = env_cols + ae_cols

    meta = pd.read_csv(BASE / "development_metadata.csv").set_index("sample_id")
    feats = pd.read_pickle(BASE / "models" / "development_features.pkl")
    meta = meta.loc[meta.index.intersection(feats.index)].copy()
    X = feats.loc[meta.index, cols].astype(float)

    # Block assignment (label semantics kept separate).
    block = np.where(meta["source"].eq("Austria_monitoring"), "survey", "po")
    meta["block"] = block
    meta["y_po"] = np.where(meta["source"].eq("iNaturalist"), 1.0,
                            np.where(meta["source"].eq("iNaturalist_other_Culicidae_background"), 0.0, np.nan))
    meta["n_events"] = meta["usable_events"].fillna(0).astype(int)
    meta["k_events"] = meta["detected_events"].fillna(0).astype(int)
    meta["y_survey"] = np.where(meta["block"].eq("survey"), (meta["k_events"] > 0).astype(float), np.nan)

    pts = pd.read_pickle(BASE / "point_features.pkl")
    ne = pd.read_csv(EXTERNAL).set_index("sample_id")
    ne = ne.loc[ne.index.intersection(pts.index)]
    neon_X = pts.loc[ne.index, cols].astype(float)

    contexts = json.loads((BASE / "models" / "contexts.json").read_text(encoding="utf-8"))
    for name, frame in (("development", X), ("NEON", neon_X)):
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"{name} features contain non-finite values")
    return Cohort(meta, X, ne, neon_X, env_cols, ae_cols, contexts)


NEON25 = WORKSPACE / "data" / "derived" / "benchmark_refinement_2026_11_05" / "heldout_neon2025"


def load_neon2025() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """NEON 2025 annual sites, their features and usable trap events.

    Events are joined to annual sites through ``namedLocation``; the join is
    checked against the published per-site usable-event counts and labels.
    """
    sites = pd.read_csv(NEON25 / "annual_sites.csv").set_index("sample_id")
    ev = pd.read_csv(NEON25 / "event_decisions.csv")
    ev = ev[ev["usable"].astype(bool)].copy()
    loc2sid = {}
    for sid, locs in sites["location_ids"].items():
        for loc in str(locs).split(";"):
            loc2sid[loc.strip()] = sid
    ev["sample_id"] = ev["namedLocation"].map(loc2sid)
    ev = ev.dropna(subset=["sample_id"])
    agg = ev.groupby("sample_id").agg(n=("target", "size"), k=("target", "sum"))
    chk = sites.join(agg)
    if not ((chk["n"] == chk["usable_events"]).all() and ((chk["k"] > 0).astype(int) == chk["target"]).all()):
        raise ValueError("NEON 2025 event join does not reproduce annual site labels")
    pts = pd.read_pickle(BASE / "point_features.pkl")
    proto = _protocol()
    cols = list(proto["representations"]["Environment41"]) + list(proto["representations"]["AlphaEarth64"])
    X = pts.loc[sites.index, cols].astype(float)
    # Same completeness rule as the NEON-2026 external table: no imputation.
    complete = np.isfinite(X.to_numpy()).all(axis=1)
    sites, X = sites.loc[complete], X.loc[complete]
    ev = ev[ev["sample_id"].isin(sites.index)]
    return sites, X, ev[["sample_id", "trap_hours", "target"]].reset_index(drop=True)


def neon_cluster_folds(clusters_pos: dict[str, bool], seed: int, k: int = 3) -> dict[str, int]:
    """Deal positive clusters round-robin first, then the rest, after a seeded shuffle."""
    rng = np.random.default_rng(seed)
    pos = sorted(c for c, p in clusters_pos.items() if p)
    neg = sorted(c for c, p in clusters_pos.items() if not p)
    rng.shuffle(pos)
    rng.shuffle(neg)
    return {c: i % k for i, c in enumerate(pos + neg)}


def context_split(cohort: Cohort, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks over ``cohort.meta`` for a frozen outer context."""
    idx = cohort.meta.index
    tr = idx.isin(ctx["train_ids"])
    te = idx.isin(ctx["test_ids"])
    if (tr & te).any():
        raise ValueError(f"{ctx['id']}: train/test overlap")
    return tr, te


def grid_groups(lon: np.ndarray, lat: np.ndarray, deg: float = 5.0) -> np.ndarray:
    gx = np.floor((np.asarray(lon) + 180) / deg).astype(int)
    gy = np.floor((np.asarray(lat) + 90) / deg).astype(int)
    return gx * 1000 + gy
