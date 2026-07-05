"""Utility helpers for the customer segmentation project."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def setup_logging(log_file: Path | None = None) -> logging.Logger:
    """Create a configured logger for the project."""
    logger = logging.getLogger("customer_segmentation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Return the project root path."""
    return Path(__file__).resolve().parent.parent


def create_sample_dataset(output_path: Path) -> pd.DataFrame:
    """Generate a synthetic retail customer dataset when no file is provided."""
    rng = np.random.default_rng(42)
    n_rows = 500

    segments = rng.integers(0, 4, size=n_rows)
    age = np.where(
        segments == 0,
        rng.normal(28, 5, n_rows),
        np.where(
            segments == 1,
            rng.normal(41, 7, n_rows),
            np.where(segments == 2, rng.normal(54, 6, n_rows), rng.normal(35, 5, n_rows)),
        ),
    )
    annual_income = np.where(
        segments == 0,
        rng.normal(55_000, 12_000, n_rows),
        np.where(
            segments == 1,
            rng.normal(95_000, 15_000, n_rows),
            np.where(segments == 2, rng.normal(65_000, 10_000, n_rows), rng.normal(72_000, 13_000, n_rows)),
        ),
    )
    spending_score = np.where(
        segments == 0,
        rng.normal(82, 8, n_rows),
        np.where(
            segments == 1,
            rng.normal(90, 6, n_rows),
            np.where(segments == 2, rng.normal(48, 9, n_rows), rng.normal(63, 8, n_rows)),
        ),
    )

    data: dict[str, object] = {
        "CustomerID": [f"C{i:04d}" for i in range(1, n_rows + 1)],
        "Age": np.clip(age, 18, 70).round(0),
        "Annual_Income": np.clip(annual_income, 20_000, 180_000).round(0),
        "Spending_Score": np.clip(spending_score, 0, 100).round(1),
        "Total_Purchases": np.clip(rng.normal(24, 8, n_rows), 3, 80).round(0),
        "Average_Order_Value": np.clip(rng.normal(92, 28, n_rows), 20, 300).round(2),
        "Visit_Frequency": np.clip(rng.normal(8, 3, n_rows), 1, 30).round(1),
        "Loyalty_Score": np.clip(rng.normal(68, 15, n_rows), 0, 100).round(1),
        "Promotion_Response": np.clip(rng.normal(0.55, 0.2, n_rows), 0, 1).round(2),
        "Discount_Usage": np.clip(rng.normal(0.4, 0.18, n_rows), 0, 1).round(2),
        "Product_Diversity": np.clip(rng.normal(8, 2.5, n_rows), 1, 20).round(1),
        "Customer_Tenure": np.clip(rng.normal(5.5, 2.2, n_rows), 1, 20).round(1),
        "Support_Tickets": np.clip(rng.poisson(1.2, n_rows), 0, 10).astype(int),
        "Satisfaction_Score": np.clip(rng.normal(4.1, 0.8, n_rows), 1, 5).round(1),
        "Website_Visits": np.clip(rng.normal(14, 6, n_rows), 3, 60).round(0),
        "Email_Open_Rate": np.clip(rng.normal(0.63, 0.18, n_rows), 0, 1).round(2),
        "Social_Engagement": np.clip(rng.normal(0.5, 0.2, n_rows), 0, 1).round(2),
        "Cart_Abandonment_Rate": np.clip(rng.normal(0.32, 0.12, n_rows), 0, 1).round(2),
        "Return_Rate": np.clip(rng.normal(0.12, 0.08, n_rows), 0, 1).round(2),
        "Preferred_Channel": rng.choice(["Online", "Store", "Hybrid"], size=n_rows, p=[0.55, 0.2, 0.25]),
        "Region": rng.choice(["North", "South", "East", "West"], size=n_rows),
        "Device_Type": rng.choice(["Mobile", "Desktop", "Tablet"], size=n_rows),
    }

    df = pd.DataFrame(data)
    df.loc[rng.choice(n_rows, size=25, replace=False), "Annual_Income"] = np.nan
    df.loc[rng.choice(n_rows, size=20, replace=False), "Spending_Score"] = np.nan
    df.loc[rng.choice(n_rows, size=15, replace=False), "Support_Tickets"] = np.nan
    df = pd.concat([df, df.iloc[:15].copy()], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist a dictionary to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def save_text(path: Path, content: str) -> None:
    """Persist a text report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
