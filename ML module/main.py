"""
CARIVIX AI - Model Training Pipeline
=====================================

Main entry point for the complete ML training workflow.

Pipeline:
    1. Dataset Loading
    2. Data Validation
    3. Data Cleaning (Missing Values, Duplicates)
    4. Feature Engineering (Numerical, Categorical, Date features)
    5. Train-Test Split
    6. Model Training (RandomForest / GradientBoosting / XGBoost)
    7. Model Evaluation (Classification / Regression metrics)
    8. Experiment Tracking (MLflow)
    9. Model Versioning
    10. Save Model

Usage:
    python main.py

    Optional sample data generation:
    python main.py --generate-sample-data

Configuration:
    Edit config/config.yaml to change dataset, algorithm, hyperparameters, etc.
"""

import os
import sys
import json
import copy
import logging
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    setup_logger,
    load_config,
    ensure_directory,
    get_timestamp,
    load_dataframe,
    save_dataframe,
    get_dataset_name,
    get_dataset_version,
    data_profile,
)
from src.preprocess import run_preprocessing_pipeline
from src.feature_engineering import run_feature_engineering_pipeline
from src.train import (
    train_model,
    BASELINE_CLASSIFICATION_MODELS,
    BASELINE_REGRESSION_MODELS,
)
from src.predict import run_prediction_pipeline

# =============================================================================
# Constants
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DEFAULT_LOG_FILE = os.path.join(LOG_DIR, f"training_{get_timestamp()}.log")

logger = logging.getLogger("CARIVIX_AI")

# =============================================================================
# Sample Data Generator
# =============================================================================

def generate_sample_data(output_path: str) -> None:
    """
    Generate a sample dataset for testing the pipeline.

    Creates a synthetic classification dataset with numerical, categorical,
    and datetime features.

    Args:
        output_path: Path to save the generated CSV file.
    """
    logger.info("Generating sample dataset...")
    np.random.seed(42)
    n_samples = 1000

    data = {
        # Numerical features
        "age": np.random.randint(18, 80, n_samples).astype(float),
        "income": np.random.normal(50000, 15000, n_samples).round(2),
        "credit_score": np.random.randint(300, 850, n_samples),
        "loan_amount": np.random.exponential(10000, n_samples).round(2),
        "years_employed": np.random.uniform(0, 40, n_samples).round(1),

        # Categorical features
        "education": np.random.choice(
            ["High School", "Bachelor", "Master", "PhD"], n_samples,
            p=[0.3, 0.4, 0.2, 0.1]
        ),
        "employment_status": np.random.choice(
            ["Employed", "Self-Employed", "Unemployed", "Retired"], n_samples,
            p=[0.6, 0.2, 0.1, 0.1]
        ),
        "marital_status": np.random.choice(
            ["Single", "Married", "Divorced", "Widowed"], n_samples,
            p=[0.3, 0.5, 0.15, 0.05]
        ),
        "housing_type": np.random.choice(
            ["Rent", "Own", "Mortgage", "Other"], n_samples,
            p=[0.3, 0.3, 0.3, 0.1]
        ),

        # Datetime features
        "application_date": pd.date_range(
            start="2023-01-01", periods=n_samples, freq="D"
        ).strftime("%Y-%m-%d").tolist(),

        # Target (binary classification: loan approval)
        "target": np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    }

    df = pd.DataFrame(data)

    # Introduce some missing values for testing
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices, "income"] = np.nan
    df.loc[missing_indices[:20], "credit_score"] = np.nan
    df.loc[missing_indices[20:40], "education"] = np.nan

    # Introduce some duplicates
    duplicate_rows = df.sample(n=10, random_state=42)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    ensure_directory(os.path.dirname(output_path))
    df.to_csv(output_path, index=False)
    logger.info(
        "Sample dataset generated: %s (shape: %s, with missing values & duplicates)",
        output_path,
        df.shape,
    )

# =============================================================================
# Pipeline Step Implementations
# =============================================================================

def run_dataset_profile(config_path: str) -> tuple:
    """Create a data profile for the configured dataset and save it to disk."""
    config = load_config(config_path)
    dataset_path = config.get("dataset_path", "data/raw/dataset.csv")
    full_path = os.path.join(PROJECT_ROOT, dataset_path)

    if not os.path.exists(full_path):
        logger.warning("Dataset not found at '%s'. Generating sample data...", full_path)
        generate_sample_data(full_path)

    df = load_dataframe(full_path)
    profile = data_profile(df, name=os.path.basename(full_path), logger=logger)
    profile_path = os.path.join(PROJECT_ROOT, "experiments", f"data_profile_{get_timestamp()}.json")
    ensure_directory(os.path.dirname(profile_path))
    def sanitize_for_json(value):
        if isinstance(value, dict):
            return {str(k): sanitize_for_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize_for_json(item) for item in value]
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        return value

    with open(profile_path, "w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(profile), handle, indent=2)

    logger.info("Data profile saved to: %s", profile_path)
    return profile, profile_path

def compare_models(X: pd.DataFrame, y: pd.Series, config: dict, dataset_path: str) -> tuple:
    """Train several algorithms and compare their metrics."""
    algorithms = config.get("compare_algorithms", ["RandomForest", "GradientBoosting"])
    results: list[dict] = []

    for algorithm in algorithms:
        algo_config = copy.deepcopy(config)
        algo_config["algorithm"] = algorithm
        model, metrics = train_model(X, y, algo_config, dataset_path)
        result = {"algorithm": algorithm}
        result.update(metrics)
        results.append(result)

    comparison_df = pd.DataFrame(results)
    comparison_path = os.path.join(PROJECT_ROOT, "experiments", f"model_comparison_{get_timestamp()}.csv")
    save_dataframe(comparison_df, comparison_path)
    logger.info("Model comparison saved to: %s", comparison_path)

    return comparison_df, comparison_path

def export_trained_artifacts(model, metrics: dict, output_dir: str) -> tuple:
    """Export the trained model and evaluation metadata to disk."""
    ensure_directory(output_dir)
    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(model, model_path)

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump({"metrics": metrics}, handle, indent=2)

    logger.info("Exported model artifacts to: %s", output_dir)
    return model_path, metadata_path

def step_dataset_loading(config: dict) -> pd.DataFrame:
    """
    Step 1: Load the dataset.

    Args:
        config: Configuration dictionary.

    Returns:
        Loaded DataFrame.
    """
    logger.info("-" * 50)
    logger.info("STEP 1: Dataset Loading")
    logger.info("-" * 50)

    dataset_path = config.get("dataset_path", "data/raw/dataset.csv")
    full_path = os.path.join(PROJECT_ROOT, dataset_path)

    if not os.path.exists(full_path):
        logger.warning(
            "Dataset not found at '%s'. Generating sample data...", full_path
        )
        generate_sample_data(full_path)

    df = load_dataframe(full_path)
    logger.info("Dataset loaded successfully. Shape: %s", df.shape)
    logger.info("Columns: %s", df.columns.tolist())
    logger.info("Data types:\n%s", df.dtypes)

    return df

def step_data_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Validate the dataset for basic quality checks.

    Args:
        df: Input DataFrame.

    Returns:
        Validated DataFrame.
    """
    logger.info("-" * 50)
    logger.info("STEP 2: Data Validation")
    logger.info("-" * 50)

    # Check for empty dataset
    if df.empty:
        raise ValueError("Dataset is empty. Please provide a valid dataset.")

    # Check for minimum rows
    if len(df) < 10:
        raise ValueError(
            f"Dataset has only {len(df)} rows. Minimum 10 rows required."
        )

    # Check data types
    numeric_count = len(df.select_dtypes(include=[np.number]).columns)
    categorical_count = len(df.select_dtypes(include=["object", "category"]).columns)

    logger.info("Validation checks passed:")
    logger.info("  - Dataset is not empty (rows: %d)", len(df))
    logger.info("  - Numeric columns: %d", numeric_count)
    logger.info("  - Categorical columns: %d", categorical_count)
    logger.info("  - Missing values: %d", df.isnull().sum().sum())
    logger.info("  - Duplicate rows: %d", df.duplicated().sum())

    return df

def step_data_cleaning(df: pd.DataFrame, config: dict, target_column: str):
    """
    Step 3 & 4: Data cleaning (missing values, duplicates) and preprocessing.

    Args:
        df: Input DataFrame.
        config: Configuration dictionary.
        target_column: Name of the target column.

    Returns:
        Tuple of (processed features DataFrame, target Series, transformers dict).
    """
    logger.info("-" * 50)
    logger.info("STEP 3 & 4: Data Cleaning & Preprocessing")
    logger.info("-" * 50)

    X, y, transformers = run_preprocessing_pipeline(df, config, target_column)

    logger.info("Preprocessed features shape: %s", X.shape)
    logger.info("Target distribution:\n%s", y.value_counts())

    return X, y, transformers

def step_feature_engineering(X: pd.DataFrame, y: pd.Series, config: dict):
    """
    Step 5: Feature engineering.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        config: Configuration dictionary.

    Returns:
        Tuple of (engineered features DataFrame, metadata dict).
    """
    logger.info("-" * 50)
    logger.info("STEP 5: Feature Engineering")
    logger.info("-" * 50)

    X_engineered, fe_metadata = run_feature_engineering_pipeline(X, y, config)

    logger.info(
        "Feature engineering complete. Shape: %s -> %s",
        X.shape,
        X_engineered.shape,
    )

    return X_engineered, fe_metadata

def step_train_test_split():
    """
    Step 6: Train-test split (handled inside train_model).
    """
    logger.info("-" * 50)
    logger.info("STEP 6: Train-Test Split (handled in train_model)")
    logger.info("-" * 50)

def step_model_training(X: pd.DataFrame, y: pd.Series, config: dict):
    """
    Step 7 & 8 & 9: Model training, evaluation, and MLflow tracking.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        config: Configuration dictionary.

    Returns:
        Tuple of (trained model, evaluation metrics).
    """
    logger.info("-" * 50)
    logger.info("STEP 7, 8 & 9: Model Training, Evaluation & MLflow Tracking")
    logger.info("-" * 50)

    dataset_path = os.path.join(PROJECT_ROOT, config.get("dataset_path", "data/raw/dataset.csv"))
    model, metrics = train_model(X, y, config, dataset_path)

    return model, metrics

def step_save_model():
    """
    Step 10: Save model (handled inside train_model).
    """
    logger.info("-" * 50)
    logger.info("STEP 10: Save Model (handled by train_model)")
    logger.info("-" * 50)

# =============================================================================
# Main Pipeline Orchestrator
# =============================================================================

def run_pipeline(
    config_path: str = CONFIG_PATH,
    tune: bool = False,
    compare: bool = False,
    profile: bool = False,
    export: bool = False,
) -> None:
    """
    Execute the complete ML training pipeline.

    Args:
        config_path: Path to the configuration YAML file.
        tune: Enable hyperparameter tuning for the run.
        compare: Compare multiple algorithms instead of training one.
        profile: Generate a data profile before training.
        export: Export the trained artifacts after training.
    """
    # --- Setup Logging ---
    ensure_directory(LOG_DIR)
    setup_logger(
        name="CARIVIX_AI",
        log_file=DEFAULT_LOG_FILE,
    )

    logger.info("=" * 70)
    logger.info("  CARIVIX AI - MODEL TRAINING PIPELINE")
    logger.info("  Started at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 70)

    try:
        # --- Load Configuration ---
        logger.info("Loading configuration from: %s", config_path)
        config = load_config(config_path)
        target_column = config.get("target_column", "target")

        if tune:
            config.setdefault("hyperparameter_tuning", {})["enabled"] = True
            logger.info("Hyperparameter tuning enabled via CLI flag.")

        if profile:
            run_dataset_profile(config_path)

        logger.info("Algorithm: %s", config.get("algorithm"))
        logger.info("Target column: %s", target_column)

        # --- Step 1: Dataset Loading ---
        df = step_dataset_loading(config)

        # Save raw data reference
        raw_data_path = os.path.join(PROJECT_ROOT, "data", "raw", "dataset.csv")
        if os.path.exists(raw_data_path):
            logger.info("Raw data available at: %s", raw_data_path)

        # --- Step 2: Data Validation ---
        df = step_data_validation(df)

        # --- Step 3 & 4: Data Cleaning & Preprocessing ---
        X, y, transformers = step_data_cleaning(df, config, target_column)

        # Save processed data
        processed_path = os.path.join(
            PROJECT_ROOT, "data", "processed", f"processed_{get_timestamp()}.csv"
        )
        processed_data = X.copy()
        processed_data[target_column] = y
        save_dataframe(processed_data, processed_path)

        # --- Step 5: Feature Engineering ---
        X_engineered, fe_metadata = step_feature_engineering(X, y, config)

        # --- Step 6: Train-Test Split ---
        step_train_test_split()

        if compare:
            logger.info("Running model comparison workflow...")
            comparison_df, comparison_path = compare_models(X_engineered, y, config, os.path.join(PROJECT_ROOT, config.get("dataset_path", "data/raw/dataset.csv")))
            logger.info("Comparison results: %s", comparison_df.to_string(index=False))
        else:
            # --- Step 7, 8, 9: Training, Evaluation, MLflow Tracking ---
            model, metrics = step_model_training(X_engineered, y, config)

            if export:
                export_dir = os.path.join(PROJECT_ROOT, "exports", get_timestamp())
                export_trained_artifacts(model, metrics, export_dir)

            # --- Step 10: Save Model (handled inside train_model) ---
            step_save_model()

        # --- Final Summary ---
        logger.info("=" * 70)
        logger.info("  PIPELINE EXECUTION COMPLETE")
        logger.info("=" * 70)
        if compare:
            logger.info("Comparison report saved to: %s", os.path.join(PROJECT_ROOT, "experiments"))
        else:
            logger.info("Final Metrics:")
            for metric_name, metric_value in metrics.items():
                logger.info("  %s: %.4f", metric_name, metric_value)
        logger.info("")
        logger.info("Model saved in: models/")
        logger.info("Experiment logs in: experiments/")
        logger.info("Training logs in: logs/")
        logger.info("=" * 70)

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        logger.error("=" * 70)
        logger.error("  PIPELINE FAILED")
        logger.error("=" * 70)
        sys.exit(1)

# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """
    CLI entry point for the CARIVIX AI Model Training Pipeline.

    Supports:
        python main.py                    # Run the full pipeline
        python main.py --generate-sample-data  # Generate sample data only
    """
    parser = argparse.ArgumentParser(
        description="CARIVIX AI - Model Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --generate-sample-data
  python main.py --config path/to/config.yaml
        """,
    )

    parser.add_argument(
        "--generate-sample-data",
        action="store_true",
        help="Generate a sample dataset and save to data/raw/dataset.csv",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_PATH,
        help=f"Path to configuration YAML file (default: {CONFIG_PATH})",
    )

    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        metavar="DATA_PATH",
        help="Run prediction on a dataset using the latest trained model",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning for the current run",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train and compare multiple algorithms instead of a single run",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Generate a data profile before training",
    )

    parser.add_argument(
        "--export",
        action="store_true",
        help="Export the trained model artifacts after training",
    )

    args = parser.parse_args()

    # Setup logging
    ensure_directory(LOG_DIR)
    setup_logger(name="CARIVIX_AI", log_file=DEFAULT_LOG_FILE)

    if args.generate_sample_data:
        # Generate sample data only
        sample_path = os.path.join(PROJECT_ROOT, "data", "raw", "dataset.csv")
        generate_sample_data(sample_path)
        logger.info("Sample data generated. You can now run: python main.py")

    elif args.predict:
        # Run prediction pipeline
        config = load_config(args.config)
        model_save_path = os.path.join(
            PROJECT_ROOT, config.get("model_save_path", "models/")
        )
        from src.predict import load_latest_model

        model, model_path = load_latest_model(model_save_path)
        data_path = os.path.join(PROJECT_ROOT, args.predict)

        output_path = os.path.join(
            PROJECT_ROOT,
            "data",
            "processed",
            f"predictions_{get_timestamp()}.csv",
        )

        results = run_prediction_pipeline(
            model_path=model_path,
            data_path=data_path,
            output_path=output_path,
            include_proba=True,
            config_path=args.config,
        )
        logger.info("Predictions saved to: %s", output_path)

    else:
        # Run the full training pipeline
        run_pipeline(
            config_path=args.config,
            tune=args.tune,
            compare=args.compare,
            profile=args.profile,
            export=args.export,
        )

if __name__ == "__main__":
    main()


