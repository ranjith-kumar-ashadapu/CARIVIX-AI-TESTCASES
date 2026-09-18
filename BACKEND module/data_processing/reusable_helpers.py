"""
CARIVIX AI - Data Processing Engine
Reusable helper functions for cleaning, transforming, and structuring
raw data into analysis-ready datasets.
"""

import logging
from typing import List, Dict, Any, Optional, Sequence

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("carivix.data_processing")


def remove_duplicates(df: pd.DataFrame, subset: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Drop exact or subset-based duplicate rows, logging how many were removed."""
    before = len(df)
    cleaned = df.drop_duplicates(subset=subset)
    logger.info("Removed %s duplicate rows", before - len(cleaned))
    return cleaned.reset_index(drop=True)


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "drop",
    fill_value: Any = None,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Handle missing values with a chosen strategy.

    strategy: "drop" | "fill" | "mean" | "median" | "ffill" | "bfill"
    columns: restrict the operation to specific columns (default: all).
    """
    target_cols = list(columns) if columns else df.columns.tolist()
    df = df.copy()

    if strategy == "drop":
        return df.dropna(subset=target_cols).reset_index(drop=True)
    elif strategy == "fill":
        df[target_cols] = df[target_cols].fillna(fill_value)
    elif strategy == "mean":
        for col in target_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
    elif strategy == "median":
        for col in target_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
    elif strategy in ("ffill", "bfill"):
        df[target_cols] = df[target_cols].fillna(method=strategy)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and snake_case all column names for consistency."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def normalize_numeric_column(df: pd.DataFrame, column: str, method: str = "minmax") -> pd.DataFrame:
    """
    Normalize a numeric column for use in ML models / forecasting inputs.

    method: "minmax" (0-1 scaling) or "zscore" (standardization).
    """
    df = df.copy()
    values = df[column].astype(float)

    if method == "minmax":
        min_v, max_v = values.min(), values.max()
        df[f"{column}_norm"] = (values - min_v) / (max_v - min_v) if max_v != min_v else 0.0
    elif method == "zscore":
        mean, std = values.mean(), values.std()
        df[f"{column}_norm"] = (values - mean) / std if std != 0 else 0.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return df


def detect_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.Series:
    """
    Return a boolean Series flagging outliers in `column` using the IQR rule.
    Useful before feeding data into forecasting/prediction models.
    """
    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return (df[column] < lower) | (df[column] > upper)


def add_time_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Feature-engineer standard calendar features (year, month, day, weekday,
    quarter) from a date column, for use in time-series / forecasting models.
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df[f"{date_column}_year"] = df[date_column].dt.year
    df[f"{date_column}_month"] = df[date_column].dt.month
    df[f"{date_column}_day"] = df[date_column].dt.day
    df[f"{date_column}_weekday"] = df[date_column].dt.weekday
    df[f"{date_column}_quarter"] = df[date_column].dt.quarter
    return df


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of dict records (e.g. from an API) into a clean DataFrame."""
    return pd.DataFrame.from_records(records)


def save_to_parquet(df: pd.DataFrame, path: str) -> None:
    """Persist a cleaned dataset to the data lake / warehouse layer as Parquet."""
    df.to_parquet(path, index=False)
    logger.info("Saved %s rows to %s", len(df), path)


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Quick data-quality profile: shape, missing-value counts, and dtypes.
    Handy for a pre-processing sanity check / QA report.
    """
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_by_column": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
