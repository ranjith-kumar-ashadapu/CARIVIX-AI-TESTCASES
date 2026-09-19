"""
Utility functions for the CARIVIX AI Model Training pipeline.

Provides:
- Logging setup
- Configuration loading
- Helper utilities
- Timer context manager for performance profiling
- Data profiling for quick EDA
- Retry decorator for fault-tolerant operations
- Exception handling decorators
"""

import os
import sys
import time
import logging
import functools
import traceback
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml
import pandas as pd
import numpy as np

# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logger(
    name: str = "CARIVIX_AI",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up and return a logger with console and optional file handlers.

    Args:
        name: Logger name.
        log_file: Path to log file (optional).
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # --- File Handler (optional) ---
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger

# =============================================================================
# Configuration Loading
# =============================================================================

def resolve_path(path: str, search_dirs: Optional[List[str]] = None) -> str:
    """
    Resolve a file path with fallback search locations.

    Relative paths are first checked as-is from the current working directory,
    then against any provided search directories.
    """
    if os.path.isabs(path):
        return path

    if os.path.exists(path):
        return os.path.abspath(path)

    if search_dirs is None:
        search_dirs = []

    for base_dir in search_dirs:
        candidate = os.path.abspath(os.path.join(base_dir, path))
        if os.path.exists(candidate):
            return candidate

    return os.path.abspath(path)

def load_config(config_path: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.
        base_dir: Optional directory to resolve relative paths against.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    logger = logging.getLogger("CARIVIX_AI")

    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Search in: current working directory, project root, and the config/ subdirectory
    config_subdir = os.path.join(base_dir, "config")
    config_path = resolve_path(config_path, search_dirs=[os.getcwd(), base_dir, config_subdir])

    if not os.path.exists(config_path):
        default_config = os.path.join(base_dir, "config", "config.yaml")
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"  Please ensure the file exists. Try using the default config:\n"
            f"    python main.py --config config/config.yaml\n"
            f"  Or run without --config to use the default:\n"
            f"    python main.py"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file)
            logger.info("Configuration loaded successfully from %s", config_path)
            return config
        except yaml.YAMLError as exc:
            logger.error("Failed to parse YAML config: %s", exc)
            raise

# =============================================================================
# Timer Context Manager
# =============================================================================

@contextmanager
def Timer(name: str = "Operation", logger: Optional[logging.Logger] = None):
    """
    Context manager for timing code blocks.

    Usage:
        with Timer("Data Loading"):
            df = load_dataframe("data.csv")

    Args:
        name: Name of the operation being timed.
        logger: Logger instance. If None, uses default CARIVIX_AI logger.
    """
    if logger is None:
        logger = logging.getLogger("CARIVIX_AI")

    start_time = time.time()
    logger.info("⏱️  %s started...", name)
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        if elapsed >= 60:
            logger.info("⏱️  %s completed in %.2f minutes", name, elapsed / 60)
        else:
            logger.info("⏱️  %s completed in %.2f seconds", name, elapsed)

# =============================================================================
# Retry Decorator
# =============================================================================

def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple = (Exception,),
    logger: Optional[logging.Logger] = None,
) -> Callable:
    """
    Decorator that retries a function on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        delay: Initial delay between retries in seconds.
        backoff_factor: Multiplier for delay after each retry.
        exceptions: Tuple of exceptions to catch and retry on.
        logger: Logger instance.

    Returns:
        Decorated function with retry logic.

    Usage:
        @retry_on_failure(max_retries=3, delay=2.0)
        def load_data():
            return pd.read_csv("data.csv")
    """
    if logger is None:
        logger = logging.getLogger("CARIVIX_AI")

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    logger.warning(
                        "Attempt %d/%d failed for '%s': %s",
                        attempt,
                        max_retries,
                        func.__name__,
                        exc,
                    )
                    if attempt < max_retries:
                        logger.info(
                            "Retrying in %.2f seconds...", current_delay
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor

            logger.error(
                "All %d attempts failed for '%s'. Last error: %s",
                max_retries,
                func.__name__,
                last_exception,
            )
            raise last_exception

        return wrapper

    return decorator

# =============================================================================
# Data Profiling
# =============================================================================

def data_profile(
    df: pd.DataFrame,
    name: str = "Dataset",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Generate a comprehensive data profile for quick EDA.

    Analyzes:
    - Shape, columns, data types
    - Missing values per column
    - Unique values per column
    - Basic statistics for numeric columns
    - Value counts for categorical columns
    - Correlation warnings

    Args:
        df: Input DataFrame.
        name: Name of the dataset for logging.
        logger: Logger instance.

    Returns:
        Dictionary containing profile information.
    """
    if logger is None:
        logger = logging.getLogger("CARIVIX_AI")

    logger.info("=" * 60)
    logger.info("DATA PROFILE: %s", name)
    logger.info("=" * 60)

    profile = {
        "name": name,
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {},
        "missing_values": {},
        "missing_pct": {},
        "unique_values": {},
        "numeric_stats": {},
        "categorical_stats": {},
        "warnings": [],
    }

    # --- Basic Info ---
    logger.info("Shape: %s", df.shape)
    logger.info("Columns: %d", len(df.columns))

    # --- Data Types ---
    for col, dtype in df.dtypes.items():
        profile["dtypes"][col] = str(dtype)
    logger.info("Data types:\n%s", df.dtypes)

    # --- Missing Values ---
    total_cells = df.size
    total_missing = df.isnull().sum().sum()
    missing_pct = (total_missing / total_cells) * 100
    profile["total_missing"] = int(total_missing)
    profile["missing_percentage"] = round(missing_pct, 2)

    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        profile["missing_values"][col] = missing_count
        profile["missing_pct"][col] = round((missing_count / len(df)) * 100, 2)

    if total_missing > 0:
        logger.info("Missing values: %d / %d (%.2f%%)", total_missing, total_cells, missing_pct)
        cols_with_missing = {col: cnt for col, cnt in profile["missing_values"].items() if cnt > 0}
        logger.info("Columns with missing values: %s", cols_with_missing)
        if missing_pct > 20:
            profile["warnings"].append(f"High missing value ratio: {missing_pct:.2f}%")

    # --- Unique Values ---
    for col in df.columns:
        profile["unique_values"][col] = int(df[col].nunique())

    # --- Numeric Statistics ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        logger.info("Numeric columns (%d):", len(numeric_cols))
        for col in numeric_cols:
            stats = {
                "mean": round(df[col].mean(), 4) if not df[col].isnull().all() else None,
                "std": round(df[col].std(), 4) if not df[col].isnull().all() else None,
                "min": round(df[col].min(), 4) if not df[col].isnull().all() else None,
                "25%": round(df[col].quantile(0.25), 4) if not df[col].isnull().all() else None,
                "50%": round(df[col].quantile(0.50), 4) if not df[col].isnull().all() else None,
                "75%": round(df[col].quantile(0.75), 4) if not df[col].isnull().all() else None,
                "max": round(df[col].max(), 4) if not df[col].isnull().all() else None,
                "skewness": round(df[col].skew(), 4) if not df[col].isnull().all() else None,
            }
            profile["numeric_stats"][col] = stats
            logger.debug("  %s: mean=%.2f, std=%.2f, min=%.2f, max=%.2f", col, stats["mean"], stats["std"], stats["min"], stats["max"])

            # Check for high skewness
            if stats["skewness"] is not None and abs(stats["skewness"]) > 1.0:
                profile["warnings"].append(f"Column '{col}' has high skewness: {stats['skewness']:.2f}")

            # Check for constant columns
            if stats["std"] is not None and stats["std"] == 0:
                profile["warnings"].append(f"Column '{col}' is constant (std=0)")

    # --- Categorical Statistics ---
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        logger.info("Categorical columns (%d):", len(categorical_cols))
        for col in categorical_cols:
            value_counts = df[col].value_counts().to_dict()
            # Convert numpy types to native Python types
            value_counts = {str(k): int(v) for k, v in value_counts.items()}
            profile["categorical_stats"][col] = {
                "unique": int(df[col].nunique()),
                "top_values": dict(list(value_counts.items())[:5]),
                "top_value": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
            }
            logger.debug("  %s: %d unique values", col, df[col].nunique())

            # Check for high cardinality
            if df[col].nunique() > 50:
                profile["warnings"].append(f"Column '{col}' has high cardinality: {df[col].nunique()} unique values")

    # --- Duplicate Rows ---
    duplicate_count = int(df.duplicated().sum())
    profile["duplicate_rows"] = duplicate_count
    if duplicate_count > 0:
        logger.info("Duplicate rows: %d", duplicate_count)
        duplicate_pct = (duplicate_count / len(df)) * 100
        if duplicate_pct > 10:
            profile["warnings"].append(f"High duplicate ratio: {duplicate_pct:.2f}%")

    # --- Summary ---
    logger.info("Profile warnings: %d", len(profile["warnings"]))
    for warning in profile["warnings"]:
        logger.warning("  [warning] %s", warning)

    logger.info("=" * 60)
    return profile

# =============================================================================
# Helper Utilities
# =============================================================================

def ensure_directory(path: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path to ensure exists.
    """
    os.makedirs(path, exist_ok=True)
    logger = logging.getLogger("CARIVIX_AI")
    logger.debug("Directory ensured: %s", path)

def get_timestamp() -> str:
    """
    Get current timestamp as a formatted string suitable for file naming.

    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_dataframe(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """
    Save a pandas DataFrame to CSV.

    Args:
        df: DataFrame to save.
        path: Destination file path.
        index: Whether to write row indices.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
    logger = logging.getLogger("CARIVIX_AI")
    logger.info("DataFrame saved to %s (shape: %s)", path, df.shape)

def load_dataframe(
    path: str,
    file_type: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load a data file into a pandas DataFrame.

    Supports CSV, Excel (.xlsx, .xls), and other pandas-readable formats.

    Args:
        path: Path to data file.
        file_type: Force file type ('csv', 'excel', 'parquet'). If None, inferred from extension.
        **kwargs: Additional arguments passed to pandas reader.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is unsupported.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    if file_type is None:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            file_type = "csv"
        elif ext in (".xlsx", ".xls"):
            file_type = "excel"
        elif ext == ".parquet":
            file_type = "parquet"
        else:
            file_type = "csv"  # Default fallback

    logger = logging.getLogger("CARIVIX_AI")

    try:
        if file_type == "csv":
            df = pd.read_csv(path, **kwargs)
        elif file_type == "excel":
            df = pd.read_excel(path, **kwargs)
        elif file_type == "parquet":
            df = pd.read_parquet(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: '{file_type}'")

        logger.info("DataFrame loaded from %s (shape: %s)", path, df.shape)
        return df

    except Exception as exc:
        logger.error("Failed to load data from %s: %s", path, exc)
        raise

def get_dataset_name(filepath: str) -> str:
    """
    Extract dataset name from file path.

    Args:
        filepath: Path to dataset file.

    Returns:
        Dataset name (filename without extension).
    """
    return os.path.splitext(os.path.basename(filepath))[0]

def get_dataset_version(filepath: str) -> str:
    """
    Generate a dataset version based on file modification time.

    Args:
        filepath: Path to dataset file.

    Returns:
        Version string.
    """
    if os.path.exists(filepath):
        mod_time = os.path.getmtime(filepath)
        return datetime.fromtimestamp(mod_time).strftime("%Y%m%d_%H%M%S")
    return "v1.0.0"

