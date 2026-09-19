"""
Experiment Tracking Module for CARIVIX AI Model Training pipeline.

Manages experiment records across all baseline model runs:
- Saves results to experiments.csv (tabular format)
- Saves results to model_metrics.json (JSON format)
- Tracks: model name, hyperparameters, dataset, training time, metrics, timestamp
- Identifies the best-performing model based on primary metric
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from src.utils import ensure_directory, get_timestamp

logger = logging.getLogger("CARIVIX_AI")

# =============================================================================
# Constants
# =============================================================================

EXPERIMENTS_DIR = "experiments"
EXPERIMENTS_CSV = os.path.join(EXPERIMENTS_DIR, "experiments.csv")
MODEL_METRICS_JSON = os.path.join(EXPERIMENTS_DIR, "model_metrics.json")

# =============================================================================
# Serialization Helpers
# =============================================================================

def _serialize_value(value: Any) -> Any:
    """
    Recursively serialize numpy/pandas types to native Python types for JSON/CSV.

    Args:
        value: Value to serialize.

    Returns:
        JSON-serializable value.
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    return value

# =============================================================================
# Core Experiment Tracking Functions
# =============================================================================

def save_experiment_record(
    record: Dict[str, Any],
    experiments_dir: str = EXPERIMENTS_DIR,
) -> None:
    """
    Save an experiment record to both experiments.csv and model_metrics.json.

    Args:
        record: Dictionary containing experiment data with keys:
            - timestamp: Run timestamp
            - model_name: Algorithm/model name
            - hyperparameters: Model hyperparameters dict
            - dataset_name: Name of the dataset
            - dataset_version: Dataset version
            - task_type: 'classification' or 'regression'
            - training_time_seconds: Training duration
            - metrics: Dictionary of evaluation metrics
            - model_path: Path to saved model file
            - is_best: Whether this is the best model so far
    """
    ensure_directory(experiments_dir)
    csv_path = os.path.join(experiments_dir, "experiments.csv")
    json_path = os.path.join(experiments_dir, "model_metrics.json")

    # Serialize record
    record_serialized = _serialize_value(record)
    record_serialized["timestamp"] = str(record_serialized.get("timestamp", ""))
    record_serialized["hyperparameters"] = json.dumps(
        record_serialized.get("hyperparameters", {}), default=str
    )
    record_serialized["metrics"] = json.dumps(
        record_serialized.get("metrics", {}), default=str
    )

    # --- Append to CSV ---
    df_record = pd.DataFrame([record_serialized])
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            combined = pd.concat([existing_df, df_record], ignore_index=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            combined = df_record
    else:
        combined = df_record
    combined.to_csv(csv_path, index=False)
    logger.info("Experiment record appended to: %s", csv_path)

    # --- Append to JSON ---
    record_json = _serialize_value(record)
    record_json["timestamp"] = str(record_json.get("timestamp", ""))
    record_json["hyperparameters"] = record_json.get("hyperparameters", {})
    record_json["metrics"] = record_json.get("metrics", {})

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                existing_json = json.load(handle)
            if not isinstance(existing_json, list):
                existing_json = [existing_json]
        except (json.JSONDecodeError, FileNotFoundError):
            existing_json = []
    else:
        existing_json = []

    existing_json.append(record_json)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(existing_json, handle, indent=2, default=str)
    logger.info("Experiment record appended to: %s", json_path)

def load_experiments_csv(
    experiments_dir: str = EXPERIMENTS_DIR,
) -> pd.DataFrame:
    """
    Load the experiments.csv file into a DataFrame.

    Args:
        experiments_dir: Directory containing experiments.csv.

    Returns:
        DataFrame with experiment records, or empty DataFrame if file doesn't exist.
    """
    csv_path = os.path.join(experiments_dir, "experiments.csv")
    if not os.path.exists(csv_path):
        logger.warning("No experiments CSV found at: %s", csv_path)
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        logger.info("Loaded %d experiment records from: %s", len(df), csv_path)
        return df
    except Exception as exc:
        logger.error("Failed to load experiments CSV: %s", exc)
        return pd.DataFrame()

def load_model_metrics_json(
    experiments_dir: str = EXPERIMENTS_DIR,
) -> List[Dict[str, Any]]:
    """
    Load the model_metrics.json file.

    Args:
        experiments_dir: Directory containing model_metrics.json.

    Returns:
        List of experiment record dicts, or empty list if file doesn't exist.
    """
    json_path = os.path.join(experiments_dir, "model_metrics.json")
    if not os.path.exists(json_path):
        logger.warning("No model metrics JSON found at: %s", json_path)
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            logger.info("Loaded %d model metrics records from: %s", len(data), json_path)
            return data
        return [data]
    except Exception as exc:
        logger.error("Failed to load model metrics JSON: %s", exc)
        return []

# =============================================================================
# Best Model Identification
# =============================================================================

def get_primary_metric(task_type: str) -> str:
    """
    Get the primary evaluation metric for a given task type.

    Args:
        task_type: 'classification' or 'regression'.

    Returns:
        Primary metric name.
    """
    if task_type == "classification":
        return "f1_score"
    return "r2_score"

def is_higher_better(metric_name: str) -> bool:
    """
    Determine if a higher value is better for a given metric.

    Args:
        metric_name: Name of the metric.

    Returns:
        True if higher is better, False if lower is better.
    """
    higher_better_metrics = {
        "accuracy", "precision", "recall", "f1_score", "roc_auc",
        "r2_score", "cv_mean", "cv_std",
    }
    lower_better_metrics = {
        "mae", "mse", "rmse", "training_time_seconds",
    }
    if metric_name in higher_better_metrics:
        return True
    if metric_name in lower_better_metrics:
        return False
    # Default: assume higher is better
    return True

def find_best_model(
    experiments_dir: str = EXPERIMENTS_DIR,
    task_type: str = "classification",
    metric: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Find the best-performing model from experiment records.

    Args:
        experiments_dir: Directory containing experiment files.
        task_type: 'classification' or 'regression'.
        metric: Primary metric to compare. If None, uses default for task type.

    Returns:
        Dict with best model info, or None if no records found.
    """
    if metric is None:
        metric = get_primary_metric(task_type)

    records = load_model_metrics_json(experiments_dir)
    if not records:
        logger.warning("No experiment records found to determine best model.")
        return None

    higher_better = is_higher_better(metric)
    best_record = None
    best_value = None

    for record in records:
        metrics = record.get("metrics", {})
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except (json.JSONDecodeError, TypeError):
                continue

        value = metrics.get(metric)
        if value is None:
            continue

        try:
            value = float(value)
        except (ValueError, TypeError):
            continue

        if best_value is None:
            best_value = value
            best_record = record
        elif higher_better and value > best_value:
            best_value = value
            best_record = record
        elif not higher_better and value < best_value:
            best_value = value
            best_record = record

    if best_record is not None:
        logger.info(
            "Best model found: %s (%s = %.4f)",
            best_record.get("model_name", "Unknown"),
            metric,
            best_value,
        )
        best_record["is_best"] = True
        best_record["best_metric"] = metric
        best_record["best_value"] = best_value

    return best_record

# =============================================================================
# Summary Generation
# =============================================================================

def generate_comparison_table(
    experiments_dir: str = EXPERIMENTS_DIR,
    task_type: str = "classification",
) -> pd.DataFrame:
    """
    Generate a comparison table of all baseline models.

    Args:
        experiments_dir: Directory containing experiment files.
        task_type: 'classification' or 'regression'.

    Returns:
        DataFrame with model comparison data.
    """
    records = load_model_metrics_json(experiments_dir)
    if not records:
        return pd.DataFrame()

    rows = []
    for record in records:
        metrics = record.get("metrics", {})
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except (json.JSONDecodeError, TypeError):
                metrics = {}

        row = {
            "Model": record.get("model_name", "Unknown"),
            "Dataset": record.get("dataset_name", ""),
            "Training Time (s)": record.get("training_time_seconds", 0),
            "Timestamp": record.get("timestamp", ""),
            "Model Path": record.get("model_path", ""),
        }
        # Add metrics dynamically
        for metric_name, metric_value in metrics.items():
            try:
                row[metric_name] = round(float(metric_value), 4)
            except (ValueError, TypeError):
                row[metric_name] = metric_value

        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    # Sort by primary metric descending
    primary_metric = get_primary_metric(task_type)
    if primary_metric in comparison_df.columns:
        comparison_df = comparison_df.sort_values(
            by=primary_metric, ascending=False
        ).reset_index(drop=True)

    logger.info("Generated comparison table with %d models.", len(comparison_df))
    return comparison_df

