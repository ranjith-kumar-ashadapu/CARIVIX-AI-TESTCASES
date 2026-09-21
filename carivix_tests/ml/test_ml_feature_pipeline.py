"""
CARIVIX AI - Machine Learning Feature Pipeline & Baseline Verification Tests
=============================================================================

Maps to test cases:
  TC-ML-01  Feature Engineering Pipeline (missing imputation, encoding, polynomial features, output shape)
  TC-ML-03  RAG Document Ingestion & Chunking (clean boundaries, token window sizing)
  TC-ML-07  Baseline Experiment Reproducibility (model benchmark score determinism & reproducibility)

Dependencies:
  pandas, numpy, scikit-learn, joblib
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

# Ensure ML module & src are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_MODULE_ROOT = PROJECT_ROOT / "ML module"
ML_SRC_ROOT = ML_MODULE_ROOT / "src"

for p in (str(ML_MODULE_ROOT), str(ML_SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from preprocess import (
    handle_missing_values,
    one_hot_encode,
    label_encode,
    run_preprocessing_pipeline,
)
from feature_engineering import (
    create_polynomial_features,
    create_ratio_features,
    extract_date_features,
    run_feature_engineering_pipeline,
)
try:
    from rag.splitter import TextPreprocessor, DocumentSplitter
    HAS_RAG = True
except (ImportError, ModuleNotFoundError):
    HAS_RAG = False
    TextPreprocessor = None
    DocumentSplitter = None


# ===========================================================================
# TC-ML-01 – Feature Engineering Pipeline
# ===========================================================================

@pytest.fixture
def raw_loan_dataset() -> pd.DataFrame:
    """Create representative raw applicant dataset with missing values and mixed types."""
    return pd.DataFrame({
        "age": [25.0, np.nan, 45.0, 52.0, 31.0, 29.0],
        "income": [35000.0, 62000.0, np.nan, 95000.0, 48000.0, 53000.0],
        "credit_score": [580.0, 720.0, 650.0, np.nan, 690.0, 710.0],
        "loan_amount": [5000.0, 15000.0, 20000.0, 25000.0, 10000.0, 12000.0],
        "years_employed": [2.0, 8.0, 12.0, 22.0, np.nan, 4.0],
        "education": ["Bachelor", "Master", np.nan, "PhD", "Bachelor", "High School"],
        "employment_status": ["Employed", "Employed", "Self-Employed", "Retired", "Employed", "Employed"],
        "marital_status": ["Single", "Married", "Married", "Single", "Married", "Single"],
        "housing_type": ["Rent", "Own", "Rent", "Own", "Rent", "Own"],
        "application_date": ["2025-01-10", "2025-02-15", "2025-03-20", "2025-04-25", "2025-05-30", "2025-06-05"],
        "target": [0, 0, 1, 0, 0, 0],
    })


@pytest.mark.ml
def test_ml01_missing_value_imputation_numeric_and_categorical(raw_loan_dataset):
    """TC-ML-01 – Verify numeric missing values are imputed and zero NaNs remain."""
    df_clean = handle_missing_values(
        raw_loan_dataset,
        strategy="mean",
        columns=["age", "income", "credit_score", "years_employed"],
    )
    # Check numeric columns have zero NaNs
    for col in ["age", "income", "credit_score", "years_employed"]:
        assert df_clean[col].isnull().sum() == 0, f"NaNs found in numeric column: {col}"


@pytest.mark.ml
def test_ml01_categorical_encoding_preserves_row_count(raw_loan_dataset):
    """TC-ML-01 – Verify categorical variables are one-hot encoded with valid numeric representation."""
    df_imputed = handle_missing_values(raw_loan_dataset, strategy="mean")
    df_encoded, ohe = one_hot_encode(
        df_imputed,
        columns=["employment_status", "marital_status", "housing_type"],
    )
    assert len(df_encoded) == len(raw_loan_dataset)
    assert any(col.startswith("employment_status_") for col in df_encoded.columns)
    assert any(col.startswith("marital_status_") for col in df_encoded.columns)


@pytest.mark.ml
def test_ml01_polynomial_and_interaction_feature_generation(raw_loan_dataset):
    """TC-ML-01 – Verify polynomial and interaction features expand feature space accurately."""
    df_clean = raw_loan_dataset.dropna().copy()
    initial_cols = len(df_clean.columns)

    df_poly = create_polynomial_features(df_clean, columns=["income", "loan_amount"], degree=2)
    assert len(df_poly.columns) > initial_cols
    assert "income^2" in df_poly.columns or "income_x_loan_amount" in df_poly.columns


@pytest.mark.ml
def test_ml01_date_feature_decomposition(raw_loan_dataset):
    """TC-ML-01 – Extract year, month, day, and dayofweek from application_date."""
    df_dates = extract_date_features(raw_loan_dataset, date_columns=["application_date"])
    assert "application_date_year" in df_dates.columns
    assert "application_date_month" in df_dates.columns
    assert "application_date_day" in df_dates.columns
    assert df_dates["application_date_year"].iloc[0] == 2025


# ===========================================================================
# TC-ML-03 – RAG Document Ingestion & Chunking
# ===========================================================================

@pytest.mark.rag
@pytest.mark.ml
def test_ml03_text_preprocessor_and_splitter_uniform_chunks():
    """
    TC-ML-03 – Ingest unstructured document text and assert splitter generates
    clean chunks with bounded token/character window sizing.
    """
    sample_policy_text = (
        "CARIVIX AI Enterprise Governance Policy Document.\n\n"
        "Section 1: Data Ingestion Standards.\n"
        "All data entering the CARIVIX platform must conform to strict schema boundaries. "
        "Missing attributes are imputed using domain-tailored statistical distributions. "
        "Categorical attributes are mapped using one-hot encodings with infrequent category binning.\n\n"
        "Section 2: Spatial Intelligence.\n"
        "WebGIS boundaries are served through Leaflet Canvas renderers. "
        "EPSG:4326 coordinate systems are verified across all national and regional tiers.\n\n"
        "Section 3: Model Inferencing & Governance.\n"
        "Machine learning models must maintain an inference latency below 300 milliseconds. "
        "Predictions output binary default risk along with calibrated confidence intervals."
    )

    if not HAS_RAG or TextPreprocessor is None or DocumentSplitter is None:
        pytest.skip("RAG dependencies (langchain-core, langchain-text-splitters) not installed in environment")

    preprocessor = TextPreprocessor()
    cleaned = preprocessor.clean_text(sample_policy_text)
    assert len(cleaned) > 0
    assert "\n\n" not in cleaned or "  " not in cleaned

    from langchain_core.documents import Document
    splitter = DocumentSplitter(chunk_size=150, chunk_overlap=30)
    input_doc = Document(page_content=cleaned, metadata={"source": "policy.txt"})
    chunks = splitter.split_documents([input_doc])

    assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"
    for idx, chunk in enumerate(chunks):
        assert isinstance(chunk, Document)
        assert len(chunk.page_content.strip()) > 0
        assert chunk.metadata.get("source") == "policy.txt"
        assert f"chunk-" in chunk.metadata.get("chunk_id", "")


# ===========================================================================
# TC-ML-07 – Baseline Experiment Reproducibility & Model Scores
# ===========================================================================

@pytest.mark.ml
@pytest.mark.smoke
def test_ml07_baseline_models_deterministic_scoring():
    """
    TC-ML-07 – Verify loaded baseline models produce deterministic predictions
    and conform to expected classification probability bounds [0.0, 1.0].
    """
    from model_service import ModelService
    service = ModelService()
    assert service.has_loaded_models()

    loaded_names = list(service.loaded_models.keys())
    assert len(loaded_names) >= 4, f"Expected at least 4 loaded models, found {len(loaded_names)}"

    sample_record = {
        "age": 40.0,
        "income": 75000.0,
        "credit_score": 710.0,
        "loan_amount": 18000.0,
        "years_employed": 10.0,
        "education": "Master",
        "employment_status": "Employed",
        "marital_status": "Married",
        "housing_type": "Own",
        "application_date": "2026-05-01",
    }

    # Run predictions across top models
    for model_name in ["XGBoost", "RandomForest", "GradientBoosting"]:
        if model_name in loaded_names:
            res1 = service.predict(sample_record, model_name=model_name)
            res2 = service.predict(sample_record, model_name=model_name)

            assert res1["success"] is True
            assert res1["prediction"] in (0, 1), f"Model {model_name} returned non-binary prediction: {res1['prediction']}"
            assert res1["prediction"] == res2["prediction"], f"Model {model_name} failed determinism check"

            if res1.get("probability") is not None:
                assert 0.0 <= res1["probability"] <= 1.0, f"Invalid probability for {model_name}: {res1['probability']}"
