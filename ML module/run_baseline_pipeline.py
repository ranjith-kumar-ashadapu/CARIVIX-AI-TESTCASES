#!/usr/bin/env python3
"""
CARIVIX AI - Baseline Model Pipeline Entry Point
=================================================

Trains multiple baseline ML models on the prepared dataset,
evaluates them, tracks experiments, saves models, and tests
the complete end-to-end data flow.

Usage:
    python run_baseline_pipeline.py
    python run_baseline_pipeline.py --config config/config.yaml
    python run_baseline_pipeline.py --data data/processed/processed_20260730_125704.csv
    python run_baseline_pipeline.py --target target
    python run_baseline_pipeline.py --verbose
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logger, ensure_directory, get_timestamp
from src.baseline_pipeline import run_baseline_pipeline
from src.model_registry import get_model_summary
from src.experiment_tracking import (
    load_experiments_csv,
    load_model_metrics_json,
    get_primary_metric,
)

# =============================================================================
# Constants
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DEFAULT_LOG_FILE = os.path.join(LOG_DIR, f"baseline_{get_timestamp()}.log")

logger = logging.getLogger("CARIVIX_AI")

# =============================================================================
# Display Utilities
# =============================================================================

def print_header(title: str, char: str = "=", width: int = 70) -> None:
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")

def print_data_flow_status(data_flow_status: Dict[str, str]) -> None:
    """Print the data flow status in a readable format."""
    print_header("END-TO-END DATA FLOW STATUS", "-")
    print(f"{'Pipeline Stage':<30} {'Status':<10}")
    print("-" * 40)
    for stage, status in data_flow_status.items():
        print(f"{stage:<30} {status:<10}")
    print("-" * 40)

def print_best_model(best_model: Optional[Dict[str, Any]], task_type: str) -> None:
    """Print the best-performing model details."""
    print_header("BEST-PERFORMING MODEL")

    if best_model is None:
        print("No model records found.")
        return

    primary_metric = get_primary_metric(task_type)
    metrics = best_model.get("metrics", {})
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except (json.JSONDecodeError, TypeError):
            metrics = {}

    best_value = best_model.get("best_value", metrics.get(primary_metric, "N/A"))

    print(f"  Model Name:           {best_model.get('model_name', 'Unknown')}")
    print(f"  Primary Metric:       {primary_metric} = {best_value}")
    print(f"  Dataset:              {best_model.get('dataset_name', 'Unknown')}")
    print(f"  Training Time (s):    {best_model.get('training_time_seconds', 'N/A')}")
    print(f"  Model Path:           {best_model.get('model_path', 'N/A')}")
    print(f"  Timestamp:            {best_model.get('timestamp', 'N/A')}")

    if metrics:
        print(f"\n  All Metrics:")
        for metric_name, metric_value in metrics.items():
            try:
                print(f"    {metric_name:<25}: {float(metric_value):.4f}")
            except (ValueError, TypeError):
                print(f"    {metric_name:<25}: {metric_value}")

def print_comparison_table(comparison_table: pd.DataFrame) -> None:
    """Print the model comparison table."""
    print_header("MODEL COMPARISON TABLE")

    if comparison_table.empty:
        print("No model comparison data available.")
        return

    # Select relevant columns for display
    display_cols = [c for c in ["Model", "accuracy", "precision", "recall", "f1_score", "roc_auc",
                                 "mae", "mse", "rmse", "r2_score",
                                 "Training Time (s)", "Dataset"]
                    if c in comparison_table.columns]

    if not display_cols:
        display_cols = comparison_table.columns.tolist()

    table_data = comparison_table[display_cols]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 30)
    print(table_data.to_string(index=False))
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
    pd.reset_option("display.max_colwidth")

def print_evaluation_metrics(comparison_table: pd.DataFrame) -> None:
    """Print evaluation metrics for all models."""
    print_header("EVALUATION METRICS SUMMARY")

    if comparison_table.empty:
        print("No metrics data available.")
        return

    # Extract metric columns (everything except Model, Dataset, Timestamp, Model Path, Training Time)
    exclude_cols = {"Model", "Dataset", "Timestamp", "Model Path", "Training Time (s)"}
    metric_cols = [c for c in comparison_table.columns if c not in exclude_cols]

    if not metric_cols:
        print("No metric columns found in comparison data.")
        return

    for model_name in comparison_table["Model"].unique():
        model_data = comparison_table[comparison_table["Model"] == model_name]
        print(f"\n  {model_name}:")
        for col in metric_cols:
            if col in model_data.columns:
                try:
                    val = model_data[col].iloc[0]
                    print(f"    {col:<25}: {float(val):.4f}")
                except (ValueError, TypeError):
                    print(f"    {col:<25}: {val}")
        train_time_col = "Training Time (s)"
        if train_time_col in model_data.columns:
            print(f"    {'Training Time (s)':<25}: {model_data[train_time_col].iloc[0]}")

def print_training_times(all_results: List[Dict[str, Any]]) -> None:
    """Print training time for each model."""
    print_header("TRAINING TIME PER MODEL")

    if not all_results:
        print("No training results available.")
        return

    print(f"{'Model':<30} {'Training Time (s)':<20} {'Status':<15}")
    print("-" * 65)
    for result in all_results:
        model_name = result.get("model_name", "Unknown")
        train_time = result.get("training_time_seconds", 0)
        status = "✓ Success" if result.get("model") is not None else f"✗ {result.get('error', 'Failed')[:30]}"
        print(f"{model_name:<30} {train_time:<20.4f} {status:<15}")
    print("-" * 65)

def print_saved_model_location() -> None:
    """Print the model directory and experiment tracking file locations."""
    print_header("SAVED ARTIFACTS LOCATIONS")

    models_dir = os.path.join(PROJECT_ROOT, "models")
    experiments_dir = os.path.join(PROJECT_ROOT, "experiments")
    experiments_csv = os.path.join(experiments_dir, "experiments.csv")
    model_metrics_json = os.path.join(experiments_dir, "model_metrics.json")

    print(f"  Models Directory:          {models_dir}")
    print(f"  Experiment CSV:            {experiments_csv}")
    print(f"  Model Metrics JSON:        {model_metrics_json}")

    # Count saved models
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
        print(f"  Saved Models:              {len(model_files)} files")

    # Count experiment records
    if os.path.exists(experiments_csv):
        try:
            df = pd.read_csv(experiments_csv)
            print(f"  Experiment Records:        {len(df)} records")
        except Exception:
            pass

def print_final_confirmation(success: bool) -> None:
    """Print the final data flow test confirmation."""
    print_header("END-TO-END DATA FLOW TEST")

    if success:
        print("  ✅  DATA FLOW TEST EXECUTED SUCCESSFULLY")
        print("")
        print("  Complete Pipeline Flow:")
        print("    Dataset → Data Validation → Preprocessing → Feature Engineering →")
        print("    Train/Test Split → Model Training → Prediction → Evaluation →")
        print("    Experiment Tracking → Save Model")
        print("")
        print("  All stages completed without errors.")
    else:
        print("  ❌  DATA FLOW TEST FAILED")
        print("  Check the logs above for error details.")

# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main entry point for the baseline model pipeline."""
    parser = argparse.ArgumentParser(
        description="CARIVIX AI - Baseline Model Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_baseline_pipeline.py
  python run_baseline_pipeline.py --config config/config.yaml
  python run_baseline_pipeline.py --data data/processed/processed_20260730_125704.csv
  python run_baseline_pipeline.py --target target
  python run_baseline_pipeline.py --verbose
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file (default: config/config.yaml)",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to the prepared dataset CSV (overrides config)",
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Name of the target column (overrides config)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    # --- Setup Logging ---
    ensure_directory(LOG_DIR)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(
        name="CARIVIX_AI",
        log_file=DEFAULT_LOG_FILE,
        level=log_level,
    )

    logger.info("=" * 70)
    logger.info("  CARIVIX AI - BASELINE MODEL PIPELINE")
    logger.info("  Started at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 70)

    # --- Resolve config path ---
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)

    # --- Run Pipeline ---
    print_header("BASELINE MODEL TRAINING & DATA FLOW TEST")
    print(f"  Configuration: {config_path}")

    results = run_baseline_pipeline(
        config_path=config_path,
        data_path=args.data,
        target_column=args.target,
    )

    # --- Display Results ---
    print("\n")
    print("=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)

    # 1. Data Flow Status
    print_data_flow_status(results.get("data_flow_status", {}))

    if results.get("success", False):
        # 2. Best Model
        print_best_model(
            results.get("best_model"),
            results.get("task_type", "classification"),
        )

        # 3. Comparison Table
        comparison_table = results.get("comparison_table", pd.DataFrame())
        if not comparison_table.empty:
            print_comparison_table(comparison_table)
            print_evaluation_metrics(comparison_table)

        # 4. Training Times
        print_training_times(results.get("all_results", []))

        # 5. Saved Model Locations
        print_saved_model_location()

        # 6. Final Confirmation
        print_final_confirmation(True)

        # Save comparison table to CSV
        if not comparison_table.empty:
            comparison_path = os.path.join(
                PROJECT_ROOT, "experiments", f"baseline_comparison_{get_timestamp()}.csv"
            )
            ensure_directory(os.path.dirname(comparison_path))
            comparison_table.to_csv(comparison_path, index=False)
            print(f"\n  Comparison table saved to: {comparison_path}")

        # Save summary report
        summary_path = os.path.join(
            PROJECT_ROOT, "experiments", f"baseline_summary_{get_timestamp()}.json"
        )
        summary_data = {
            "status": "success",
            "timestamp": get_timestamp(),
            "task_type": results.get("task_type", ""),
            "dataset_name": results.get("dataset_name", ""),
            "best_model": results.get("best_model", {}).get("model_name", "") if results.get("best_model") else "",
            "data_flow_status": results.get("data_flow_status", {}),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        print(f"  Summary report saved to: {summary_path}")

    else:
        print_final_confirmation(False)
        if results.get("data_flow_status"):
            failed_stages = [
                stage for stage, status in results["data_flow_status"].items()
                if "✗" in status or "FAILED" in status
            ]
            if failed_stages:
                print(f"\n  Failed stages: {', '.join(failed_stages)}")
        if results.get("error"):
            print(f"\n  Error: {results['error']}")

    print("\n")
    logger.info("=" * 70)
    logger.info("  BASELINE PIPELINE COMPLETED")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()

