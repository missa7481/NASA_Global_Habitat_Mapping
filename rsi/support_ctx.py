"""Data-support context for a set of query cells relative to a training set:
geographic distance, sampling density and environmental novelty."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree, NearestNeighbors


def geo_tree(D, tr):
    return BallTree(np.radians(np.c_[D.lat[tr], D.lon[tr]]), metric="haversine")


def support_context(D, tr, q, env_basis=None, k_env=10, exclude_self=False):
    """Returns dict of per-query-cell arrays: dist_km, dens250 (training cells within 250 km), env_nov."""
    T = geo_tree(D, tr)
    Q = np.radians(np.c_[D.lat[q], D.lon[q]])
    d, _ = T.query(Q, k=2 if exclude_self else 1)
    dist = d[:, -1] * 6371.0
    cnt = T.query_radius(Q, r=250 / 6371.0, count_only=True) - int(exclude_self)
    if env_basis is None:
        env_basis = env_space(D)
    E = env_basis
    nn = NearestNeighbors(n_neighbors=k_env + int(exclude_self)).fit(E[tr])
    de, _ = nn.kneighbors(E[q])
    if exclude_self:
        de = de[:, 1:]
    # reference scale: typical within-training kNN distance
    ref_d, _ = nn.kneighbors(E[tr][:: max(1, len(tr) // 2000)])
    ref = np.median(ref_d[:, 1:].mean(1))
    return {"dist_km": dist, "dens250": cnt.astype(float), "env_nov": de.mean(1) / ref}


def env_space(D, n=8):
    """Climate + landscape PCA space (the environmental niche axes, excluding human and embedding)."""
    cols = D.cols["Climate"] + D.cols["Landscape"]
    return PCA(n, random_state=0).fit_transform(D.X[:, cols])
