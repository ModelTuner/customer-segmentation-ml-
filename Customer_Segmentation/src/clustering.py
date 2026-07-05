"""Clustering routines for customer segmentation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import joblib  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score as _silhouette_score  # pyright: ignore[reportUnknownVariableType]

silhouette_score = cast(Any, _silhouette_score)


class CustomerClusterer:
    """Train PCA and K-Means models for customer segmentation."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the clustering class."""
        self.logger = logger or logging.getLogger(__name__)

    def reduce_dimensions(
        self,
        features: np.ndarray,
        n_components: int = 2,
    ) -> tuple[PCA, np.ndarray]:
        """
        Reduce feature dimensions using PCA.

        Args:
            features: Scaled feature matrix.
            n_components: Number of PCA components.

        Returns:
            Tuple containing fitted PCA model and transformed data.
        """

        if features.size == 0:
            raise ValueError("Input feature matrix cannot be empty.")

        if n_components < 1:
            raise ValueError("n_components must be greater than 0.")

        self.logger.info("Applying PCA...")

        pca = PCA(
            n_components=n_components,
            random_state=42,
        )

        transformed = pca.fit_transform(features)

        self.logger.info(
            "PCA completed successfully."
        )

        self.logger.info(
            "Explained Variance Ratio: %s",
            np.round(pca.explained_variance_ratio_, 4),
        )

        return pca, transformed

    def find_optimal_clusters(
        self,
        features: np.ndarray,
        max_clusters: int = 10,
    ) -> dict[str, Any]:
        """
        Determine the optimal number of clusters using
        Elbow Method and Silhouette Score.
        """

        if features.size == 0:
            raise ValueError("Input feature matrix cannot be empty.")

        if len(features) < 2:
            raise ValueError("At least two samples are required.")

        if max_clusters < 2:
            raise ValueError("max_clusters must be at least 2.")

        inertias: list[float] = []
        silhouette_scores: list[float] = []

        self.logger.info("Searching for the optimal number of clusters...")

        for n_clusters in range(2, max_clusters + 1):

            model = KMeans(
                n_clusters=n_clusters,
                n_init=20,
                random_state=42,
            )

            labels = model.fit_predict(features)

            inertias.append(float(model.inertia_))

            score = cast(float, silhouette_score(features, labels))

            silhouette_scores.append(score)

            self.logger.info(
                "Clusters: %d | Silhouette Score: %.4f",
                n_clusters,
                score,
            )

        optimal_k = int(np.argmax(silhouette_scores) + 2)

        self.logger.info(
            "Optimal number of clusters selected: %d",
            optimal_k,
        )

        return {
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "optimal_k": optimal_k,
        }

    def fit_kmeans(
        self,
        features: np.ndarray,
        n_clusters: int,
        save_model: bool = True,
    ) -> tuple[KMeans, np.ndarray]:
        """
        Train the final K-Means model.

        Args:
            features: PCA transformed data.
            n_clusters: Number of clusters.
            save_model: Save trained model.

        Returns:
            Trained model and predicted labels.
        """

        if n_clusters < 2:
            raise ValueError("Number of clusters must be at least 2.")

        self.logger.info(
            "Training K-Means model with %d clusters...",
            n_clusters,
        )

        model = KMeans(
            n_clusters=n_clusters,
            n_init=20,
            random_state=42,
        )

        labels = model.fit_predict(features)

        if save_model:

            output_dir = Path("outputs/models")
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            model_path = output_dir / "kmeans_model.pkl"

            cast(Any, joblib).dump(model, model_path)

            self.logger.info(
                "Model saved to %s",
                model_path,
            )

        self.logger.info("K-Means training completed successfully.")

        return model, labels


if __name__ == "__main__":
    print("=" * 60)
    print("Customer Segmentation - Clustering Module")
    print("=" * 60)
    print("This module contains:")
    print("✓ PCA Dimensionality Reduction")
    print("✓ Elbow Method")
    print("✓ Silhouette Score")
    print("✓ K-Means Clustering")
    print("\nRun main.py to execute the complete pipeline.")