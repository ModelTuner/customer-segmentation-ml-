"""Evaluation helpers for clustering quality and reporting."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score as _silhouette_score  # pyright: ignore[reportUnknownVariableType]

silhouette_score = cast(Any, _silhouette_score)


def summarize_clusters(
    labels: np.ndarray,
    pca_features: np.ndarray,
    pca_model: Any,
    kmeans_model: Any,
) -> dict[str, Any]:
    """Compute major clustering metrics and a compact summary."""
    cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()
    summary: dict[str, Any] = {
        "inertia": float(kmeans_model.inertia_),
        "silhouette_score": float(silhouette_score(pca_features, labels)),
        "cluster_sizes": {str(key): int(value) for key, value in cluster_sizes.items()},
        "pca_explained_variance_ratio": [float(value) for value in pca_model.explained_variance_ratio_],
    }
    return summary
