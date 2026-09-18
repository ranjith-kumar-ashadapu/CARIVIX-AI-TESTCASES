from __future__ import annotations
from typing import Any, Optional

import pandas as pd

from data_transformation import DataProcessingEngine, save_processed
from preprocessing_optimized import optimize_dtypes
from logging_config import get_logger
import os
import yaml

logger = get_logger("carivix.processing_workflow")


class ProcessingWorkflowError(Exception):
    """Raised when the processing pipeline fails to complete."""


class ProcessingWorkflow:
    """
    Single entry point connecting data cleaning, transformation, and
    performance optimization into one callable pipeline for Data Science
    workflows. Wraps DataProcessingEngine and the optimization helpers
    so calling code runs one method instead of orchestrating each step.
    """

    def __init__(self, missing_strategy: str = "median", optimize_memory: bool = True):
        self.engine = DataProcessingEngine(missing_strategy=missing_strategy)
        self.optimize_memory = optimize_memory

    def run_with_profile(self, df: pd.DataFrame, profile_name: str, config_path: Optional[str] = None) -> pd.DataFrame:
        path = config_path or os.path.join(
            os.path.dirname(__file__), "..", "config", "processing_profiles.yaml"
        )
        with open(path) as f:
            profiles = yaml.safe_load(f)

        if profile_name not in profiles:
            raise ProcessingWorkflowError(f"Unknown profile '{profile_name}'. Available: {list(profiles.keys())}")

        settings = profiles[profile_name]
        return self.run(
            df,
            dedup_subset=settings.get("dedup_subset"),
            normalize_cols=settings.get("normalize_cols"),
            date_col=settings.get("date_col"),
            categorical_cols=settings.get("categorical_cols"),
        )

    def run(
        self,
        df: pd.DataFrame,
        *,
        dedup_subset: Optional[list] = None,
        normalize_cols: Optional[list] = None,
        date_col: Optional[str] = None,
        categorical_cols: Optional[list] = None,
        schema: Optional[dict] = None,
        save_path: Optional[str] = None,
        save_fmt: str = "parquet",
    ) -> pd.DataFrame:
        if df is None:
            raise ProcessingWorkflowError("No data provided: df cannot be None")
        if not isinstance(df, pd.DataFrame):
            raise ProcessingWorkflowError(f"Expected a pandas DataFrame, got {type(df).__name__}")
        if len(df) == 0:
            raise ProcessingWorkflowError("Cannot process an empty DataFrame")
        try:
            logger.info(f"Starting processing workflow on {len(df)} row(s)", extra={"context": "workflow"})
            result = self.engine.run(
                df,
                dedup_subset=dedup_subset,
                normalize_cols=normalize_cols,
                date_col=date_col,
                categorical_cols=categorical_cols,
                schema=schema,
            )

            if self.optimize_memory:
                result = optimize_dtypes(result)
                logger.info("Applied memory optimization to output", extra={"context": "workflow"})

            if save_path:
                save_processed(result, save_path, fmt=save_fmt)
                logger.info(f"Saved workflow output to {save_path}", extra={"context": "workflow"})

            logger.info(f"Workflow complete. Final shape: {result.shape}", extra={"context": "workflow"})
            return result

        except Exception as exc:
            logger.error(f"Processing workflow failed: {exc}", extra={"context": "workflow"})
            raise ProcessingWorkflowError(f"Workflow failed: {exc}") from exc