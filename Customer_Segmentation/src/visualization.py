"""Visualization helpers for customer segmentation."""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import matplotlib

if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]

plt = cast(Any, plt)
sns = cast(Any, sns)


class Visualizer:
    """Create plots for exploratory analysis and clustering results."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def _display_plot(self) -> None:
        """Display the current figure and wait for the user to close it."""
        if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
            return

        plt.show(block=True)

    def plot_missing_values(self, data: pd.DataFrame) -> None:
        """Plot a missing-value heatmap."""
        plt.figure(figsize=(14, 6))
        sns.heatmap(data.isna(), cbar=False, cmap="viridis")
        plt.title("Missing Value Heatmap")
        plt.tight_layout()
        plt.savefig(self.output_dir / "missing_value_heatmap.png", dpi=300)
        self._display_plot()
        plt.close()

    def plot_correlation_heatmap(self, data: pd.DataFrame) -> None:
        """Plot a correlation heatmap for numeric columns."""
        numeric_data = data.select_dtypes(include=["number"])
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            numeric_data.corr(numeric_only=True),
            annot=False,
            cmap="coolwarm",
        )
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_heatmap.png", dpi=300)
        self._display_plot()
        plt.close()

    def plot_feature_distributions(self, data: pd.DataFrame) -> None:
        """Plot distributions for the main numeric features."""
        numeric_columns = data.select_dtypes(include=["number"]).columns[:6]

        _, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for axis, column in zip(axes, numeric_columns):
            sns.histplot(data[[column]], kde=True, ax=axis)
            axis.set_title(f"Distribution of {column}")

        for axis in axes[len(numeric_columns):]:
            axis.axis("off")

        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_distributions.png", dpi=300)
        self._display_plot()
        plt.close()

    def plot_elbow_curve(
        self,
        inertias: list[float],
        max_clusters: int,
    ) -> None:
        """Plot inertia versus cluster count."""
        plt.figure(figsize=(8, 5))
        plt.plot(
            range(2, max_clusters + 1),
            inertias,
            marker="o",
            linewidth=2,
        )
        plt.title("Elbow Method for Optimal Clusters")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / "elbow_method.png", dpi=300)
        self._display_plot()
        plt.close()

    def plot_silhouette_scores(
        self,
        scores: list[float],
        max_clusters: int,
    ) -> None:
        """Plot silhouette scores versus cluster count."""
        plt.figure(figsize=(8, 5))
        plt.plot(
            range(2, max_clusters + 1),
            scores,
            marker="o",
            color="green",
            linewidth=2,
        )
        plt.title("Silhouette Scores by Cluster Count")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / "silhouette_scores.png", dpi=300)
        self._display_plot()
        plt.close()

    def plot_pca_variance(self, pca_model: Any) -> None:
        """Plot explained variance ratio for PCA components."""
        plt.figure(figsize=(8, 5))
        plt.plot(
            range(
                1,
                len(pca_model.explained_variance_ratio_) + 1,
            ),
            pca_model.explained_variance_ratio_,
            marker="o",
            linewidth=2,
        )
        plt.title("PCA Explained Variance Ratio")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "pca_explained_variance.png",
            dpi=300,
        )
        self._display_plot()
        plt.close()

    def plot_clusters(
        self,
        pca_data: Any,
        labels: list[int],
    ) -> None:
        """Create a 2D PCA scatter plot with cluster labels."""
        plot_frame = pd.DataFrame(
            pca_data,
            columns=["PC1", "PC2"],
        )

        plot_frame["Cluster"] = labels

        plt.figure(figsize=(8, 6))

        sns.scatterplot(
            data=plot_frame,
            x="PC1",
            y="PC2",
            hue="Cluster",
            palette="Set2",
            s=70,
        )

        plt.title("PCA Cluster Visualization")
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "pca_cluster_visualization.png",
            dpi=300,
        )
        self._display_plot()
        plt.close()

    def plot_cluster_distribution(
        self,
        labels: list[int],
    ) -> None:
        """Visualize the size of each cluster."""
        counts = pd.Series(labels).value_counts().sort_index()

        plt.figure(figsize=(8, 5))

        sns.barplot(
            x=counts.index.astype(str),
            y=counts.values,
            hue=counts.index.astype(str),
            palette="viridis",
            dodge=False,
            legend=False,
        )

        plt.title("Cluster Distribution")
        plt.xlabel("Cluster")
        plt.ylabel("Customer Count")
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "cluster_distribution.png",
            dpi=300,
        )
        self._display_plot()
        plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Customer Segmentation Visualization Module")
    print("=" * 60)
    print("Run main.py to generate all graphs.")