"""
CARIVIX AI - Data Processing Engine
Example integration: validation + error handling + logging working together.
Author: Sindhu Bollikonda (Python/R Developer)

Simulates a small slice of Module 2 (Data Processing Engine): pulling a raw
dataset (e.g. district-level unemployment data feeding Government/Economic
Intelligence), validating it, cleaning it, and logging every step.
"""

import pandas as pd

from data_validation import ColumnRule, DataValidator, Schema
from error_handling import DataSourceError, safe_run
from logging_config import get_logger

logger = get_logger(__name__)


@safe_run(retries=2, delay_seconds=1, exceptions=(DataSourceError,))
def fetch_raw_data() -> pd.DataFrame:
    """Stand-in for a real API/scraper call (Module 1: Data Acquisition Engine)."""
    logger.info("Fetching raw district-level dataset", extra={"context": "acquisition"})
    return pd.DataFrame(
        {
            "district": ["Hyderabad", "Warangal", "Karimnagar", None, "Hyderabad"],
            "unemployment_rate": [5.2, 7.8, 6.1, 4.9, 5.2],
            "population": [10_004_000, 811_000, 4_611_000, -100, 10_004_000],
        }
    )


def run_pipeline() -> pd.DataFrame:
    logger.info("=== CARIVIX Data Processing Pipeline started ===", extra={"context": "pipeline"})

    raw_df = fetch_raw_data()

    schema = Schema(
        columns=[
            ColumnRule(name="district", dtype=str, required=True, allow_null=False),
            ColumnRule(name="unemployment_rate", dtype=float, min_value=0, max_value=100),
            ColumnRule(name="population", dtype=int, min_value=0),
        ]
    )

    validator = DataValidator(schema)
    report = validator.validate(raw_df)
    print(report.summary())

    clean_df = validator.clean(raw_df)
    logger.info("=== Pipeline completed successfully ===", extra={"context": "pipeline"})
    return clean_df


if __name__ == "__main__":
    result_df = run_pipeline()
    print("\nFinal analysis-ready dataset:")
    print(result_df)
