"""
CARIVIX AI — Data Processing Engine (Module 2)
================================================
Implements the three core functions specified in the CARIVIX AI project doc:
    1. Data Cleaning     : missing value removal, duplicate removal, standardization
    2. Data Transformation: normalization, feature engineering, structuring
    3. Data Storage       : warehouse/data-lake-ready output (parquet/csv)

Author role: Python/R Developer (Sindhu Bollikonda)
Tasks covered: "Develop data transformation scripts" + "Optimize preprocessing functions"

Dependencies: pandas, numpy  (optionally pyarrow for parquet storage)
    pip install pandas numpy pyarrow --break-system-packages
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("carivix.data_processing")


# ----------------------------------------------------------------------
# 1. DATA CLEANING
# ----------------------------------------------------------------------
class DataCleaner:
    """Missing value removal, duplicate removal, standardization."""

    def __init__(self, missing_strategy: Literal["drop", "mean", "median", "mode", "constant"] = "median",
                 fill_value=None):
        self.missing_strategy = missing_strategy
        self.fill_value = fill_value

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names: lowercase, snake_case, no whitespace."""
        df = df.copy()
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(r"[^\w\s]", "", regex=True)
            .str.replace(r"\s+", "_", regex=True)
        )
        return df

    def standardize_text(self, df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
        """Trim whitespace and normalize casing for object/text columns."""
        df = df.copy()
        cols = columns if columns is not None else df.select_dtypes(include=["object", "string"]).columns
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
                df.loc[df[col].isin(["nan", "None", ""]), col] = np.nan
        return df

    def remove_duplicates(self, df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
        logger.info(f"Removed {before - len(df)} duplicate rows.")
        return df

    def handle_missing(self, df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
        df = df.copy()
        cols = columns if columns is not None else df.columns

        if self.missing_strategy == "drop":
            before = len(df)
            df = df.dropna(subset=cols).reset_index(drop=True)
            logger.info(f"Dropped {before - len(df)} rows with missing values.")
            return df

        for col in cols:
            if col not in df.columns or df[col].isna().sum() == 0:
                continue
            if self.missing_strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            elif self.missing_strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            elif self.missing_strategy == "mode":
                mode_val = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else self.fill_value)
            elif self.missing_strategy == "constant":
                df[col] = df[col].fillna(self.fill_value)
            else:
                # fallback for non-numeric columns under mean/median strategy
                mode_val = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "unknown")
        return df

    def coerce_numeric(self, df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
        df = df.copy()
        cols = columns if columns is not None else []
        for col in cols:
            if col in df.columns:
                before_na = df[col].isna().sum()
                df[col] = pd.to_numeric(df[col], errors="coerce")
                after_na = df[col].isna().sum()
                if after_na > before_na:
                    logger.warning(f"Column '{col}': {after_na - before_na} non-numeric value(s) converted to NaN.")
        return df

    def clean(self, df: pd.DataFrame, dedup_subset: list[str] | None = None) -> pd.DataFrame:
        """Full cleaning pipeline: standardize -> dedupe -> impute."""
        df = self.standardize_columns(df)
        df = self.standardize_text(df)
        df = self.remove_duplicates(df, subset=dedup_subset)
        df = self.handle_missing(df)
        return df


# ----------------------------------------------------------------------
# 2. DATA TRANSFORMATION
# ----------------------------------------------------------------------
class DataTransformer:
    """Normalization, feature engineering, structuring."""

    def normalize_minmax(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            c_min, c_max = df[col].min(), df[col].max()
            df[col] = 0.0 if c_max == c_min else (df[col] - c_min) / (c_max - c_min)
        return df

    def normalize_zscore(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            std = df[col].std()
            df[col] = 0.0 if std == 0 else (df[col] - df[col].mean()) / std
        return df

    def engineer_date_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Extract year/month/day/day-of-week/quarter from a date column."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[f"{date_col}_year"] = df[date_col].dt.year
        df[f"{date_col}_month"] = df[date_col].dt.month
        df[f"{date_col}_day"] = df[date_col].dt.day
        df[f"{date_col}_dow"] = df[date_col].dt.dayofweek
        df[f"{date_col}_quarter"] = df[date_col].dt.quarter
        return df

    def encode_categorical(self, df: pd.DataFrame, columns: list[str],
                            method: Literal["onehot", "label"] = "onehot") -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
        if method == "onehot":
            df = pd.get_dummies(df, columns=columns, drop_first=False, dtype=int)
        else:
            for col in columns:
                df[col] = df[col].astype("category").cat.codes
        return df

    def enforce_schema(self, df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
        """Structure the dataset to a target schema {column: dtype}, adding
        missing columns as NaN and casting types for analysis-readiness."""
        df = df.copy()
        for col, dtype in schema.items():
            if col not in df.columns:
                df[col] = np.nan
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError):
                logger.warning(f"Could not cast column '{col}' to {dtype}; left as-is.")
        return df[list(schema.keys())]


# ----------------------------------------------------------------------
# 3. DATA STORAGE (warehouse / data-lake ready output)
# ----------------------------------------------------------------------
def save_processed(df: pd.DataFrame, path: str, fmt: Literal["parquet", "csv"] = "parquet") -> None:
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} rows x {len(df.columns)} cols -> {path} ({fmt})")


# ----------------------------------------------------------------------
# ORCHESTRATOR — matches "Data Sources -> Data Collection -> Data Processing" flow
# ----------------------------------------------------------------------
@dataclass
class DataProcessingEngine:
    """End-to-end pipeline: clean -> transform -> store."""
    missing_strategy: str = "median"
    cleaner: DataCleaner = field(init=False)
    transformer: DataTransformer = field(init=False)

    def __post_init__(self):
        self.cleaner = DataCleaner(missing_strategy=self.missing_strategy)
        self.transformer = DataTransformer()

    def run(self, df: pd.DataFrame, *, dedup_subset=None,
             normalize_cols: list[str] | None = None,
             date_col: str | None = None,
             categorical_cols: list[str] | None = None,
             schema: dict[str, str] | None = None) -> pd.DataFrame:
        logger.info("Starting CARIVIX data processing pipeline...")
        df = self.cleaner.clean(df, dedup_subset=dedup_subset)
        if normalize_cols:
            df = self.cleaner.coerce_numeric(df, normalize_cols)
            df = self.cleaner.handle_missing(df, columns=normalize_cols)

        if date_col:
            df = self.transformer.engineer_date_features(df, date_col)
        if normalize_cols:
            df = self.transformer.normalize_zscore(df, normalize_cols)
        if categorical_cols:
            df = self.transformer.encode_categorical(df, categorical_cols)
        if schema:
            df = self.transformer.enforce_schema(df, schema)

        logger.info(f"Pipeline complete. Final shape: {df.shape}")
        return df


if __name__ == "__main__":
    # Minimal smoke test
    sample = pd.DataFrame({
        " Company Name ": ["Acme Inc", "Acme Inc", "Beta Co", None],
        "Revenue": [100.0, 100.0, np.nan, 250.0],
        "Region": ["South", "South", "North", "north "],
        "Report Date": ["2026-01-05", "2026-01-05", "2026-02-10", "2026-03-01"],
    })

    engine = DataProcessingEngine(missing_strategy="median")
    result = engine.run(
        sample,
        dedup_subset=None,
        normalize_cols=["revenue"],
        date_col="report_date",
        categorical_cols=["region"],
    )
    print(result)
