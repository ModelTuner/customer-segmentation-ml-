"""Run the customer segmentation pipeline end to end."""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportGeneralTypeIssues=false

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-not-found, import-untyped]
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.clustering import CustomerClusterer
from src.evaluation import summarize_clusters
from src.preprocessing import DataProcessor
from src.utils import create_sample_dataset, ensure_directory, get_project_root, save_json, save_text, setup_logging
from src.visualization import Visualizer


def build_pipeline() -> None:
    """Execute the segmentation workflow and save outputs."""
    root = get_project_root()
    data_dir = root / "data"
    outputs_dir = root / "outputs"
    plots_dir = outputs_dir / "plots"
    reports_dir = outputs_dir / "reports"
    models_dir = outputs_dir / "models"
    ensure_directory(data_dir)
    ensure_directory(plots_dir)
    ensure_directory(reports_dir)
    ensure_directory(models_dir)

    logger = setup_logging(root / "outputs" / "logs" / "pipeline.log")
    processor = DataProcessor(logger)
    clusterer = CustomerClusterer(logger)
    visualizer = Visualizer(plots_dir)

    dataset_path = data_dir / "customer_data.csv"
    if not dataset_path.exists():
        logger.info("No dataset found; generating a sample retail dataset")
        create_sample_dataset(dataset_path)

    data = processor.load_data(dataset_path)
    cleaned_data = processor.clean_data(data)

    features, feature_columns, scaler = processor.prepare_features(cleaned_data, drop_columns=["CustomerID"])

    pca_model, pca_features = clusterer.reduce_dimensions(features, n_components=2)
    search_results = clusterer.find_optimal_clusters(features, max_clusters=8)

    optimal_k = search_results["optimal_k"]
    model, labels = clusterer.fit_kmeans(features, n_clusters=optimal_k)

    visualizer.plot_missing_values(cleaned_data)
    visualizer.plot_correlation_heatmap(cleaned_data)
    visualizer.plot_feature_distributions(cleaned_data)
    visualizer.plot_elbow_curve(search_results["inertias"], 8)
    visualizer.plot_silhouette_scores(search_results["silhouette_scores"], 8)
    visualizer.plot_pca_variance(pca_model)
    visualizer.plot_clusters(pca_features, labels.tolist())
    visualizer.plot_cluster_distribution(labels.tolist())

    summary = summarize_clusters(labels, pca_features, pca_model, model)
    save_json(reports_dir / "clustering_summary.json", summary)

    model_payload: dict[str, Any] = {
        "pca_model": pca_model,
        "kmeans_model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
    }
    joblib.dump(model_payload, models_dir / "segmentation_models.joblib")

    persona_report = build_persona_report(cleaned_data, labels.tolist(), feature_columns)
    save_text(reports_dir / "persona_report.txt", persona_report)

    business_report = build_business_report(summary, persona_report)
    save_text(reports_dir / "business_insights_report.txt", business_report)

    logger.info("Pipeline completed successfully. Outputs saved in %s", outputs_dir)


def build_persona_report(data: pd.DataFrame, labels: list[int], feature_columns: list[str]) -> str:
    """Create a persona-style report summarizing each cluster."""
    frame = data.copy()
    frame["Cluster"] = labels
    report_lines: list[str] = []
    report_lines.append("Customer Persona Report")
    report_lines.append("=" * 30)

    for cluster_id in sorted(frame["Cluster"].unique()):
        cluster_data = frame[frame["Cluster"] == cluster_id]
        report_lines.append(f"\nCluster {cluster_id}")
        report_lines.append("-" * 20)
        report_lines.append(f"Customer count: {len(cluster_data)}")
        report_lines.append(
            f"Average annual income: ${cluster_data['Annual_Income'].mean():,.0f}"
        )
        report_lines.append(
            f"Average spending score: {cluster_data['Spending_Score'].mean():.1f}"
        )
        report_lines.append(
            f"Average loyalty score: {cluster_data['Loyalty_Score'].mean():.1f}"
        )
        report_lines.append(
            f"Most common channel: {cluster_data['Preferred_Channel'].mode().iloc[0]}"
        )
        report_lines.append(
            f"Most common region: {cluster_data['Region'].mode().iloc[0]}"
        )
        report_lines.append("Persona: High-value, engaged, and retention-focused")
        report_lines.append("Recommendation: Offer personalized loyalty incentives and premium support")

    return "\n".join(report_lines)


def build_business_report(summary: dict[str, object], persona_report: str) -> str:
    """Write an executive summary with business recommendations."""
    report_lines = [
        "Business Insights Report",
        "=" * 28,
        f"Silhouette score: {summary['silhouette_score']:.3f}",
        f"Inertia: {summary['inertia']:.2f}",
        f"Cluster sizes: {summary['cluster_sizes']}",
        "",
        "Recommended actions:",
        "- Prioritize high-value clusters with targeted loyalty offers.",
        "- Improve retention for low-engagement clusters through reactivation campaigns.",
        "- Personalize promotions by customer channel and region.",
        "",
        "Persona summary:",
        persona_report,
    ]
    return "\n".join(report_lines)


if __name__ == "__main__":
    build_pipeline()
