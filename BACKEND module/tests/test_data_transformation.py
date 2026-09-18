import pandas as pd
import numpy as np
from data_transformation import DataCleaner, DataTransformer, DataProcessingEngine


def test_standardize_columns_lowercases_and_snakecases():
    df = pd.DataFrame({" Company Name ": ["Acme"], "Report Date": ["2026-01-05"]})
    cleaned = DataCleaner().standardize_columns(df)
    assert "company_name" in cleaned.columns
    assert "report_date" in cleaned.columns


def test_remove_duplicates_drops_exact_matches():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    result = DataCleaner().remove_duplicates(df)
    assert len(result) == 2


def test_encode_categorical_normalizes_case_before_encoding():
    df = pd.DataFrame({"region": ["North", "north", "South"]})
    result = DataTransformer().encode_categorical(df, columns=["region"], method="onehot")

    region_cols = [c for c in result.columns if c.startswith("region_")]
    assert sorted(region_cols) == ["region_north", "region_south"]
    assert "region_North" not in result.columns


def test_normalize_zscore_produces_zero_mean():
    df = pd.DataFrame({"value": [10.0, 20.0, 30.0]})
    result = DataTransformer().normalize_zscore(df, columns=["value"])
    assert abs(result["value"].mean()) < 1e-9


def test_full_pipeline_runs_end_to_end():
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

    assert result.shape[0] == 3
    assert "region_north" in result.columns
    assert "region_south" in result.columns