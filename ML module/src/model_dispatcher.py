"""
Model Dispatcher Module for CARIVIX AI Model Training pipeline.

Automatically determines:
- Whether the problem is Classification or Regression based on target column analysis
- Provides the complete list of baseline models for each task type
- Initializes model instances with appropriate default hyperparameters

Supports the full baseline model suite:
- Classification: LogisticRegression, DecisionTree, RandomForest, SVM, KNN, NaiveBayes, GradientBoosting, XGBoost
- Regression: LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, XGBoostRegressor, KNNRegressor, SVR
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

logger = logging.getLogger("CARIVIX_AI")

# =============================================================================
# Baseline Model Registry
# =============================================================================

CLASSIFICATION_MODELS: Dict[str, Any] = {
    "LogisticRegression": LogisticRegression,
    "DecisionTree": DecisionTreeClassifier,
    "RandomForest": RandomForestClassifier,
    "SVM": SVC,
    "KNN": KNeighborsClassifier,
    "NaiveBayes": GaussianNB,
    "GradientBoosting": GradientBoostingClassifier,
}

REGRESSION_MODELS: Dict[str, Any] = {
    "LinearRegression": LinearRegression,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "KNNRegressor": KNeighborsRegressor,
    "SVR": SVR,
}

CLASSIFICATION_MODEL_NAMES: List[str] = [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "SVM",
    "KNN",
    "NaiveBayes",
    "GradientBoosting",
    "XGBoost",
]

REGRESSION_MODEL_NAMES: List[str] = [
    "LinearRegression",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "KNNRegressor",
    "SVR",
    "XGBoostRegressor",
]

# =============================================================================
# Task Type Detection
# =============================================================================

def detect_task_type(y: pd.Series) -> str:
    """
    Automatically determine whether the problem is Classification or Regression
    based on the target column properties.

    Heuristics:
    - If target is object/boolean/categorical dtype → classification
    - If target has few unique values (≤ 20) → classification
    - If target values are integers with small range → classification
    - Otherwise → regression

    Args:
        y: Target series.

    Returns:
        'classification' or 'regression'.
    """
    logger.info("Detecting task type from target column...")

    # Check dtype first
    if y.dtype == "object" or str(y.dtype) == "category" or y.dtype == bool:
        logger.info("Target dtype is '%s' → classification", y.dtype)
        return "classification"

    # Check unique values
    unique_count = y.nunique()
    total_count = len(y)

    if unique_count <= 2:
        logger.info(
            "Target has %d unique values (≤2) → classification",
            unique_count,
        )
        return "classification"

    # Check if values are integer-like with small range
    if np.issubdtype(y.dtype, np.integer):
        if unique_count <= 20:
            logger.info(
                "Target has %d unique integer values (≤20) → classification",
                unique_count,
            )
            return "classification"
        # Check value range
        value_range = y.max() - y.min()
        if value_range <= 100 and unique_count <= 50:
            logger.info(
                "Target integer range is small (%d, %d unique) → classification",
                value_range,
                unique_count,
            )
            return "classification"

    # Check unique ratio (for floats)
    unique_ratio = unique_count / total_count
    if unique_ratio > 0.1:
        logger.info(
            "Target has high unique ratio (%.2f) → regression",
            unique_ratio,
        )
        return "regression"

    # Fallback: check if values are mostly integers
    if np.issubdtype(y.dtype, np.floating):
        integer_mask = y.dropna() == y.dropna().astype(int)
        integer_ratio = integer_mask.mean()
        if integer_ratio > 0.8 and unique_count <= 50:
            logger.info(
                "Target is mostly integer-like (%.2f ratio, %d unique) → classification",
                integer_ratio,
                unique_count,
            )
            return "classification"

    logger.info("Target appears continuous → regression")
    return "regression"

# =============================================================================
# Model Initialization
# =============================================================================

def initialize_model(
    algorithm: str,
    model_params: Optional[Dict[str, Any]] = None,
    task_type: str = "classification",
) -> Any:
    """
    Initialize and return a baseline model instance.

    Args:
        algorithm: Name of the algorithm (e.g., 'RandomForest', 'LogisticRegression').
        model_params: Dictionary of hyperparameters.
        task_type: 'classification' or 'regression'.

    Returns:
        Initialized model instance.

    Raises:
        ValueError: If algorithm is not supported.
        ImportError: If XGBoost is requested but not installed.
    """
    if model_params is None:
        model_params = {}

    logger.info(
        "Initializing model: '%s' (task: %s)", algorithm, task_type
    )

    # --- XGBoost (special handling - optional dependency) ---
    if algorithm in ("XGBoost", "XGBoostRegressor"):
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost is required but not installed. Run: pip install xgboost"
            )

        if algorithm == "XGBoost":
            return xgb.XGBClassifier(
                learning_rate=model_params.get("learning_rate", 0.1),
                n_estimators=model_params.get("n_estimators", 100),
                max_depth=model_params.get("max_depth", 6),
                subsample=model_params.get("subsample", 0.8),
                colsample_bytree=model_params.get("colsample_bytree", 0.8),
                random_state=model_params.get("random_state", 42),
                eval_metric="logloss",
                use_label_encoder=False,
                n_jobs=-1,
                verbosity=0,
            )
        return xgb.XGBRegressor(
            learning_rate=model_params.get("learning_rate", 0.1),
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", 6),
            subsample=model_params.get("subsample", 0.8),
            colsample_bytree=model_params.get("colsample_bytree", 0.8),
            random_state=model_params.get("random_state", 42),
            n_jobs=-1,
            verbosity=0,
        )

    # --- Classification Models ---
    if task_type == "classification":
        if algorithm == "LogisticRegression":
            return LogisticRegression(
                C=model_params.get("C", 1.0),
                solver=model_params.get("solver", "liblinear"),
                max_iter=model_params.get("max_iter", 500),
                random_state=model_params.get("random_state", 42),
                n_jobs=-1,
            )
        if algorithm == "DecisionTree":
            return DecisionTreeClassifier(
                max_depth=model_params.get("max_depth", None),
                min_samples_split=model_params.get("min_samples_split", 2),
                min_samples_leaf=model_params.get("min_samples_leaf", 1),
                random_state=model_params.get("random_state", 42),
            )
        if algorithm == "RandomForest":
            return RandomForestClassifier(
                n_estimators=model_params.get("n_estimators", 100),
                max_depth=model_params.get("max_depth", None),
                min_samples_split=model_params.get("min_samples_split", 2),
                min_samples_leaf=model_params.get("min_samples_leaf", 1),
                random_state=model_params.get("random_state", 42),
                n_jobs=-1,
            )
        if algorithm == "SVM":
            return SVC(
                C=model_params.get("C", 1.0),
                kernel=model_params.get("kernel", "rbf"),
                probability=True,
                random_state=model_params.get("random_state", 42),
            )
        if algorithm == "KNN":
            return KNeighborsClassifier(
                n_neighbors=model_params.get("n_neighbors", 5),
                weights=model_params.get("weights", "uniform"),
                n_jobs=-1,
            )
        if algorithm == "NaiveBayes":
            return GaussianNB()
        if algorithm == "GradientBoosting":
            return GradientBoostingClassifier(
                learning_rate=model_params.get("learning_rate", 0.1),
                n_estimators=model_params.get("n_estimators", 100),
                max_depth=model_params.get("max_depth", 3),
                random_state=model_params.get("random_state", 42),
            )
        raise ValueError(
            f"Unsupported classification algorithm '{algorithm}'. "
            f"Supported: {CLASSIFICATION_MODEL_NAMES}"
        )

    # --- Regression Models ---
    if algorithm == "LinearRegression":
        return LinearRegression(n_jobs=-1)
    if algorithm == "DecisionTreeRegressor":
        return DecisionTreeRegressor(
            max_depth=model_params.get("max_depth", None),
            min_samples_split=model_params.get("min_samples_split", 2),
            min_samples_leaf=model_params.get("min_samples_leaf", 1),
            random_state=model_params.get("random_state", 42),
        )
    if algorithm == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", None),
            min_samples_split=model_params.get("min_samples_split", 2),
            min_samples_leaf=model_params.get("min_samples_leaf", 1),
            random_state=model_params.get("random_state", 42),
            n_jobs=-1,
        )
    if algorithm == "GradientBoostingRegressor":
        return GradientBoostingRegressor(
            learning_rate=model_params.get("learning_rate", 0.1),
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", 3),
            random_state=model_params.get("random_state", 42),
        )
    if algorithm == "KNNRegressor":
        return KNeighborsRegressor(
            n_neighbors=model_params.get("n_neighbors", 5),
            weights=model_params.get("weights", "uniform"),
            n_jobs=-1,
        )
    if algorithm == "SVR":
        return SVR(
            C=model_params.get("C", 1.0),
            kernel=model_params.get("kernel", "rbf"),
        )
    raise ValueError(
        f"Unsupported regression algorithm '{algorithm}'. "
        f"Supported: {REGRESSION_MODEL_NAMES}"
    )

# =============================================================================
# Utility Functions
# =============================================================================

def get_baseline_model_names(task_type: str) -> List[str]:
    """
    Get the list of baseline model names for a given task type.

    Args:
        task_type: 'classification' or 'regression'.

    Returns:
        List of model/algorithm names.
    """
    if task_type == "classification":
        return CLASSIFICATION_MODEL_NAMES.copy()
    return REGRESSION_MODEL_NAMES.copy()

def get_default_hyperparameters(algorithm: str) -> Dict[str, Any]:
    """
    Get default hyperparameters for a given algorithm.

    Args:
        algorithm: Name of the algorithm.

    Returns:
        Dictionary of default hyperparameter values.
    """
    defaults = {
        "LogisticRegression": {"C": 1.0, "solver": "liblinear", "max_iter": 500},
        "DecisionTree": {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
        "RandomForest": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
        "SVM": {"C": 1.0, "kernel": "rbf"},
        "KNN": {"n_neighbors": 5, "weights": "uniform"},
        "NaiveBayes": {},
        "GradientBoosting": {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 3},
        "XGBoost": {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8},
        "LinearRegression": {},
        "DecisionTreeRegressor": {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
        "RandomForestRegressor": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
        "GradientBoostingRegressor": {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 3},
        "KNNRegressor": {"n_neighbors": 5, "weights": "uniform"},
        "SVR": {"C": 1.0, "kernel": "rbf"},
        "XGBoostRegressor": {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8},
    }
    return defaults.get(algorithm, {}).copy()

