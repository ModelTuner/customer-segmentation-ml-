"""Data loading, cleaning, and feature preparation for segmentation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """Handle dataset loading, cleaning, and feature preparation."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def load_data(self, path: Path) -> pd.DataFrame:
        """Load the raw dataset from disk."""
        self.logger.info("Loading dataset from %s", path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")
        return pd.read_csv(path)

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates and fill missing values in a robust way."""
        cleaned = data.copy()
        self.logger.info("Initial dataset shape: %s", cleaned.shape)

        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        self.logger.info("Shape after duplicate removal: %s", cleaned.shape)

        for column in cleaned.columns:
            if pd.api.types.is_numeric_dtype(cleaned[column]):
                cleaned[column] = cleaned[column].fillna(cleaned[column].median())
            else:
                mode_value = cleaned[column].mode(dropna=True)
                if not mode_value.empty:
                    cleaned[column] = cleaned[column].fillna(mode_value.iloc[0])
                else:
                    cleaned[column] = cleaned[column].fillna("Unknown")

        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        cleaned = cleaned.fillna(0)
        self.logger.info("Missing values remaining: %s", int(cleaned.isna().sum().sum()))

        return cleaned

    def prepare_features(
        self, data: pd.DataFrame, drop_columns: list[str] | None = None
    ) -> tuple[np.ndarray, list[str], StandardScaler]:
        """Encode categorical columns and standardize numeric features."""
        working_frame = data.copy()
        if drop_columns:
            working_frame = working_frame.drop(columns=[col for col in drop_columns if col in working_frame.columns])

        categorical_columns = working_frame.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical_columns:
            working_frame = pd.get_dummies(working_frame, columns=categorical_columns, drop_first=True)

        feature_columns = [column for column in working_frame.columns if column != "CustomerID"]
        feature_frame = working_frame[feature_columns].copy()
        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0)

        scaler = StandardScaler()
        feature_array = feature_frame.to_numpy(dtype=float)
        scaled_features = scaler.fit_transform(feature_array)  # pyright: ignore[reportUnknownMemberType]
        self.logger.info("Prepared %s features for clustering", scaled_features.shape[1])
        return scaled_features, feature_frame.columns.tolist(), scaler
