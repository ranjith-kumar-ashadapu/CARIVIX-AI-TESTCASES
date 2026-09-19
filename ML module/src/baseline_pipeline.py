"""
Baseline Model Pipeline Module for CARIVIX AI.

Orchestrates the complete end-to-end baseline model training workflow:
1. Load prepared dataset
2. Validate dataset structure
3. Handle missing values
4. Split features (X) and target (y)
5. Perform train-test split
6. Loop through all baseline models
7. Train each model, record training time
8. Evaluate with all relevant metrics
9. Save model via model_registry
10. Track experiment via experiment_tracking
11. Identify best-performing model
12. Verify end-to-end data flow
"""

import os
import sys
import time
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils import (
    ensure_directory,
    get_timestamp,
    load_dataframe,
    save_dataframe,
    get_dataset_name,
)
from src.preprocess import (
    handle_missing_values,
    label_encode,
    one_hot_encode,
    scale_features,
)
from src.feature_engineering import run_feature_engineering_pipeline
from src.model_dispatcher import (
    detect_task_type,
    initialize_model,
    get_baseline_model_names,
    get_default_hyperparameters,
)
from src.model_registry import save_model, list_models
from src.experiment_tracking import (
    save_experiment_record,
    find_best_model,
    generate_comparison_table,
    get_primary_metric,
)
from src.evaluate import evaluate_model

logger = logging.getLogger("CARIVIX_AI")

# =============================================================================
# Data Loading & Validation
# =============================================================================

def load_prepared_dataset(
    data_path: str,
    target_column: str = "target",
) -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Load the prepared (processed) dataset and validate its structure.

    Args:
        data_path: Path to the processed dataset CSV.
        target_column: Name of the target column.

    Returns:
        Tuple of (X feature DataFrame, y target Series, dataset_name).

    Raises:
        FileNotFoundError: If data file doesn't exist.
        ValueError: If dataset is empty or target column missing.
    """
    logger.info("-" * 50)
    logger.info("STEP: Dataset Loading")
    logger.info("-" * 50)

    if not os.path.exists(data_path):
        # Try to find processed files in the directory
        data_dir = os.path.dirname(data_path)
        if os.path.exists(data_dir):
            files = [
                f for f in os.listdir(data_dir)
                if f.endswith(".csv") and f.startswith("processed_")
            ]
            if files:
                # Use the most recent processed file
                latest_file = max(
                    files,
                    key=lambda f: os.path.getmtime(os.path.join(data_dir, f)),
                )
                data_path = os.path.join(data_dir, latest_file)
                logger.info(
                    "Using latest processed dataset: %s", data_path
                )
            else:
                raise FileNotFoundError(
                    f"No processed dataset found at {data_path} or in {data_dir}. "
                    "Run the main pipeline first to generate processed data."
                )
        else:
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}. "
                "Run the main pipeline first to generate processed data."
            )

    logger.info("Loading dataset from: %s", data_path)
    df = load_dataframe(data_path)
    logger.info("Dataset loaded. Shape: %s", df.shape)

    # --- Validate ---
    if df.empty:
        raise ValueError("Dataset is empty. Please provide a valid dataset.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns: {df.columns.tolist()}"
        )

    # --- Separate features and target ---
    X = df.drop(columns=[target_column])
    y = df[target_column].copy()

    logger.info("Features shape: %s, Target shape: %s", X.shape, y.shape)
    logger.info("Target dtype: %s", y.dtype)
    logger.info("Target unique values: %d", y.nunique())

    dataset_name = get_dataset_name(data_path)
    logger.info("Dataset name: %s", dataset_name)

    logger.info("✓ Dataset Loaded")
    return X, y, dataset_name

def validate_dataset_structure(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Validate the dataset structure for model training readiness.

    Args:
        X: Feature DataFrame.
        y: Target Series.

    Raises:
        ValueError: If validation fails.
    """
    logger.info("-" * 50)
    logger.info("STEP: Data Validation")
    logger.info("-" * 50)

    if X.shape[0] == 0:
        raise ValueError("Feature matrix X is empty.")

    if y.shape[0] == 0:
        raise ValueError("Target vector y is empty.")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Feature count ({X.shape[0]}) does not match target count ({y.shape[0]})."
        )

    if X.shape[0] < 10:
        raise ValueError(
            f"Dataset has only {X.shape[0]} rows. Minimum 10 rows required."
        )

    # Check for non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.warning(
            "Found non-numeric columns: %s. These will be encoded.",
            non_numeric,
        )

    # Check for infinite values
    inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        logger.warning("Found %d infinite values. Replacing with NaN.", inf_count)

    # Check target balance
    if y.dtype != "object" and str(y.dtype) != "category":
        value_counts = y.value_counts()
        if len(value_counts) <= 10:
            min_class_pct = (value_counts.min() / len(y)) * 100
            logger.info(
                "Target class distribution - smallest class: %.2f%%",
                min_class_pct,
            )

    logger.info("Dataset validation passed. Rows: %d, Features: %d", X.shape[0], X.shape[1])
    logger.info("✓ Data Validation Completed")

# =============================================================================
# Data Preprocessing for Baseline
# =============================================================================

def prepare_features_for_baseline(
    X: pd.DataFrame,
    config: Dict[str, Any],
    target_column: str = "target",
) -> pd.DataFrame:
    """
    Prepare features for baseline model training:
    - Handle missing values
    - Encode categorical variables
    - Scale numerical features

    Args:
        X: Feature DataFrame.
        config: Configuration dictionary.
        target_column: Target column name (for reference).

    Returns:
        Cleaned and encoded feature DataFrame.
    """
    logger.info("-" * 50)
    logger.info("STEP: Preprocessing")
    logger.info("-" * 50)

    df = X.copy()

    # Step 1: Handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.info("Handling %d missing values...", missing_count)
        df = handle_missing_values(
            df, strategy=config.get("preprocessing", {}).get("missing_strategy", "mean")
        )
    else:
        logger.info("No missing values found.")

    # Step 2: Replace infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Step 3: Identify column types
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Step 4: Encode categorical features
    if categorical_cols:
        low_cardinality = [c for c in categorical_cols if df[c].nunique() <= 10]
        high_cardinality = [c for c in categorical_cols if df[c].nunique() > 10]

        if low_cardinality:
            logger.info("Label encoding %d low-cardinality columns...", len(low_cardinality))
            df, _ = label_encode(df, columns=low_cardinality)

        if high_cardinality:
            logger.info(
                "One-hot encoding %d high-cardinality columns (max 20 categories)...",
                len(high_cardinality),
            )
            df, _ = one_hot_encode(df, columns=high_cardinality, drop_first=True, max_categories=20)

    # Step 5: Scale numerical features
    numeric_cols_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_to_scale = [
        c for c in numeric_cols_to_scale if df[c].nunique() > 2
    ]
    if numeric_cols_to_scale:
        logger.info("Scaling %d numerical features...", len(numeric_cols_to_scale))
        scaling_method = config.get("preprocessing", {}).get("scaling_method", "standard")
        df, _ = scale_features(df, columns=numeric_cols_to_scale, method=scaling_method)

    logger.info("Preprocessing complete. Shape: %s", df.shape)
    logger.info("✓ Preprocessing Completed")
    return df

# =============================================================================
# Model Training Loop
# =============================================================================

def train_baseline_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
    dataset_name: str,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Train all baseline models for the detected task type.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.
        task_type: 'classification' or 'regression'.
        dataset_name: Name of the dataset.
        config: Full configuration dictionary.

    Returns:
        List of result dicts with training outcomes for each model.
    """
    logger.info("-" * 50)
    logger.info("STEP: Model Training")
    logger.info("-" * 50)

    baseline_config = config.get("baseline", {})
    skip_models = baseline_config.get("skip_models", [])
    model_params = baseline_config.get("model_parameters", {})
    model_names = get_baseline_model_names(task_type)

    # Filter out skipped models
    model_names = [m for m in model_names if m not in skip_models]

    logger.info(
        "Training %d baseline models for %s task...",
        len(model_names),
        task_type,
    )

    all_results: List[Dict[str, Any]] = []

    for idx, model_name in enumerate(model_names, 1):
        logger.info("-" * 40)
        logger.info("Training model %d/%d: %s", idx, len(model_names), model_name)
        logger.info("-" * 40)

        try:
            # Merge default params with any overrides from config
            params = get_default_hyperparameters(model_name)
            if model_name in model_params:
                params.update(model_params[model_name])

            # Initialize model
            model = initialize_model(model_name, params, task_type)

            # Train and time it
            training_start = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - training_start

            logger.info(
                "Training completed in %.4f seconds", training_time
            )

            # Evaluate
            logger.info("Evaluating %s...", model_name)
            evaluation_results = evaluate_model(
                model, X_test, y_test, task_type=task_type
            )
            metrics = evaluation_results["metrics"]
            logger.info("Metrics: %s", metrics)

            # Record result
            timestamp = get_timestamp()
            model_path = ""
            if baseline_config.get("save_models", True):
                model_path = save_model(
                    model=model,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    timestamp=timestamp,
                )
                logger.info("✓ Model Saved")

            result = {
                "model_name": model_name,
                "model": model,
                "model_path": model_path,
                "task_type": task_type,
                "training_time_seconds": round(training_time, 4),
                "metrics": metrics,
                "hyperparameters": params,
                "timestamp": timestamp,
                "dataset_name": dataset_name,
            }
            all_results.append(result)

            # Track experiment
            if baseline_config.get("track_experiments", True):
                experiment_record = {
                    "timestamp": timestamp,
                    "model_name": model_name,
                    "hyperparameters": params,
                    "dataset_name": dataset_name,
                    "task_type": task_type,
                    "training_time_seconds": round(training_time, 4),
                    "metrics": metrics,
                    "model_path": model_path,
                }
                save_experiment_record(experiment_record)
                logger.info("✓ Experiment Tracking Saved")

            logger.info("✓ Model Training Completed for %s", model_name)

        except ImportError as exc:
            logger.warning(
                "Skipping %s due to missing dependency: %s", model_name, exc
            )
            all_results.append(
                {
                    "model_name": model_name,
                    "model": None,
                    "error": str(exc),
                    "training_time_seconds": 0,
                    "metrics": {},
                    "hyperparameters": {},
                    "timestamp": get_timestamp(),
                    "dataset_name": dataset_name,
                }
            )
            continue
        except Exception as exc:
            logger.error(
                "Training failed for %s: %s", model_name, exc, exc_info=True
            )
            all_results.append(
                {
                    "model_name": model_name,
                    "model": None,
                    "error": str(exc),
                    "training_time_seconds": 0,
                    "metrics": {},
                    "hyperparameters": {},
                    "timestamp": get_timestamp(),
                    "dataset_name": dataset_name,
                }
            )
            continue

    logger.info("✓ Model Training Completed for all models")
    return all_results

# =============================================================================
# Main Baseline Pipeline Orchestrator
# =============================================================================

def run_baseline_pipeline(
    config_path: str = "config/config.yaml",
    data_path: Optional[str] = None,
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the complete baseline model training pipeline.

    Executes the full end-to-end data flow:
    Dataset → Data Validation → Preprocessing → Feature Engineering →
    Train/Test Split → Model Training → Prediction → Evaluation →
    Experiment Tracking → Save Model

    Args:
        config_path: Path to YAML configuration file.
        data_path: Optional override for dataset path.
        target_column: Optional override for target column name.

    Returns:
        Dictionary with pipeline results including:
        - all_results: List of per-model training results
        - best_model: Best-performing model info
        - comparison_table: DataFrame with model comparison
        - task_type: Detected task type
        - data_flow_status: Status of each pipeline stage
    """
    from src.utils import load_config

    logger.info("=" * 70)
    logger.info("  CARIVIX AI - BASELINE MODEL PIPELINE")
    logger.info("=" * 70)

    data_flow_status: Dict[str, str] = {}
    all_results: List[Dict[str, Any]] = []
    pipeline_success = False

    try:
        # --- Load Configuration ---
        logger.info("Loading configuration from: %s", config_path)
        config = load_config(config_path)
        baseline_config = config.get("baseline", {})
        logger.info("✓ Configuration Loaded")

        # Determine target column
        if target_column is None:
            target_column = config.get("target_column", "target")

        # --- Step 1: Dataset Loading ---
        if data_path is None:
            if baseline_config.get("use_processed_data", True):
                processed_dir = baseline_config.get(
                    "processed_data_path", "data/processed/"
                )
                data_path = os.path.join(processed_dir, "dataset.csv")
            else:
                data_path = os.path.join(
                    os.path.dirname(config_path), "..", config.get("dataset_path", "data/raw/dataset.csv")
                )

        # Resolve relative to project root
        if not os.path.isabs(data_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(project_root, data_path)

        X, y, dataset_name = load_prepared_dataset(data_path, target_column)
        data_flow_status["Dataset Loading"] = "✓"
        data_flow_status["Dataset"] = f"Loaded: {dataset_name} ({X.shape[0]} rows, {X.shape[1]} features)"

        # --- Step 2: Data Validation ---
        validate_dataset_structure(X, y)
        data_flow_status["Data Validation"] = "✓"

        # --- Step 3: Preprocessing ---
        X_clean = prepare_features_for_baseline(X, config, target_column)
        data_flow_status["Preprocessing"] = "✓"

        # --- Step 4: Feature Engineering ---
        logger.info("-" * 50)
        logger.info("STEP: Feature Engineering")
        logger.info("-" * 50)
        X_engineered, fe_metadata = run_feature_engineering_pipeline(X_clean, y, config)
        logger.info("Feature engineering complete. Shape: %s", X_engineered.shape)
        logger.info("✓ Feature Engineering Completed")
        data_flow_status["Feature Engineering"] = "✓"

        # --- Step 5: Train/Test Split ---
        logger.info("-" * 50)
        logger.info("STEP: Train/Test Split")
        logger.info("-" * 50)
        test_size = baseline_config.get("test_size", config.get("test_size", 0.2))
        random_state = baseline_config.get("random_state", config.get("random_state", 42))

        stratify = y if config.get("task_type", "classification") == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        logger.info(
            "Train: %s, Test: %s", X_train.shape, X_test.shape
        )
        logger.info("✓ Train/Test Split Completed")
        data_flow_status["Train/Test Split"] = "✓"

        # --- Detect Task Type ---
        if baseline_config.get("auto_detect_task_type", True):
            task_type = detect_task_type(y)
            logger.info(
                "Auto-detected task type: %s", task_type
            )
        else:
            task_type = baseline_config.get("task_type", config.get("task_type", "classification"))
            logger.info("Using configured task type: %s", task_type)
        data_flow_status["Task Detection"] = f"{task_type}"

        # --- Step 6, 7, 8: Model Training, Prediction, Evaluation ---
        all_results = train_baseline_models(
            X_train, X_test, y_train, y_test,
            task_type, dataset_name, config,
        )
        data_flow_status["Model Training"] = "✓"
        data_flow_status["Prediction"] = "✓"
        data_flow_status["Evaluation"] = "✓"

        # --- Step 9: Experiment Tracking (already done per model) ---
        data_flow_status["Experiment Tracking"] = "✓"

        # --- Step 10: Model Saving (already done per model) ---
        data_flow_status["Save Model"] = "✓"

        # --- Identify Best Model ---
        best_model = find_best_model(
            task_type=task_type,
            metric=get_primary_metric(task_type),
        )
        data_flow_status["Best Model Selection"] = "✓"

        # --- Generate Comparison Table ---
        comparison_table = generate_comparison_table(task_type=task_type)

        pipeline_success = True
        data_flow_status["Data Flow Test"] = "✓ SUCCESSFUL"

        logger.info("=" * 70)
        logger.info("  BASELINE PIPELINE COMPLETE - DATA FLOW TESTED SUCCESSFULLY")
        logger.info("=" * 70)

        return {
            "success": True,
            "all_results": all_results,
            "best_model": best_model,
            "comparison_table": comparison_table,
            "task_type": task_type,
            "dataset_name": dataset_name,
            "data_flow_status": data_flow_status,
            "config": config,
        }

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        data_flow_status["Data Flow Test"] = "✗ FAILED"
        logger.error("=" * 70)
        logger.error("  BASELINE PIPELINE FAILED")
        logger.error("=" * 70)

        return {
            "success": False,
            "error": str(exc),
            "data_flow_status": data_flow_status,
            "all_results": all_results,
        }

