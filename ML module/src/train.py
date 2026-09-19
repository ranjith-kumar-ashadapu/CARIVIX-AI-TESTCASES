"""
Model Training Module for CARIVIX AI Model Training pipeline.

Supports:
- Classification: RandomForest, GradientBoosting, XGBoost
- Regression: RandomForestRegressor, GradientBoostingRegressor, XGBoostRegressor, LinearRegression, Ridge, Lasso
- Cross-validation training
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- SHAP explainability
- MLflow experiment tracking
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
)

from src.utils import setup_logger, ensure_directory, get_timestamp, get_dataset_name, get_dataset_version
from src.evaluate import evaluate_model

logger = logging.getLogger("CARIVIX_AI")

# =============================================================================
# Model Registry
# =============================================================================

BASELINE_CLASSIFICATION_MODELS = [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "GradientBoosting",
    "XGBoost",
    "SVM",
    "KNearestNeighbors",
    "NaiveBayes",
]

BASELINE_REGRESSION_MODELS = [
    "LinearRegression",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "XGBoostRegressor",
    "KNNRegressor",
    "SVR",
]

CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression,
    "DecisionTree": DecisionTreeClassifier,
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
    "SVM": SVC,
    "KNearestNeighbors": KNeighborsClassifier,
    "NaiveBayes": GaussianNB,
}

REGRESSION_MODELS = {
    "LinearRegression": LinearRegression,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVR": SVR,
    "KNNRegressor": KNeighborsRegressor,
}

def get_model(algorithm: str, model_params: Dict[str, Any], task_type: str = "classification") -> Any:
    """
    Initialize and return the specified machine learning model.
    """
    logger.info("Initializing model: '%s' (task: %s)", algorithm, task_type)

    # XGBoost models
    if algorithm in ("XGBoost", "XGBoostRegressor"):
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost is required but not installed. Run: pip install xgboost")

        if algorithm == "XGBoost":
            return xgb.XGBClassifier(
                learning_rate=model_params.get("learning_rate_xgb", 0.1),
                n_estimators=model_params.get("n_estimators_xgb", 100),
                max_depth=model_params.get("max_depth_xgb", 6),
                subsample=model_params.get("subsample", 0.8),
                colsample_bytree=model_params.get("colsample_bytree", 0.8),
                random_state=model_params.get("random_state", 42),
                eval_metric="logloss",
                use_label_encoder=False,
                n_jobs=-1,
                probability=True,
            )
        return xgb.XGBRegressor(
            learning_rate=model_params.get("learning_rate_xgb", 0.1),
            n_estimators=model_params.get("n_estimators_xgb", 100),
            max_depth=model_params.get("max_depth_xgb", 6),
            subsample=model_params.get("subsample", 0.8),
            colsample_bytree=model_params.get("colsample_bytree", 0.8),
            random_state=model_params.get("random_state", 42),
            n_jobs=-1,
        )

    if task_type == "classification":
        if algorithm == "LogisticRegression":
            return LogisticRegression(
                solver=model_params.get("logistic_solver", "liblinear"),
                C=model_params.get("logistic_C", 1.0),
                max_iter=model_params.get("max_iter", 500),
                random_state=model_params.get("random_state", 42),
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
        if algorithm == "GradientBoosting":
            return GradientBoostingClassifier(
                learning_rate=model_params.get("learning_rate", 0.1),
                n_estimators=model_params.get("n_estimators_gb", 100),
                max_depth=model_params.get("max_depth_gb", 3),
                random_state=model_params.get("random_state", 42),
            )
        if algorithm == "SVM":
            return SVC(
                kernel=model_params.get("svm_kernel", "rbf"),
                C=model_params.get("svm_C", 1.0),
                probability=True,
                random_state=model_params.get("random_state", 42),
            )
        if algorithm == "KNearestNeighbors":
            return KNeighborsClassifier(
                n_neighbors=model_params.get("knn_neighbors", 5),
                weights=model_params.get("knn_weights", "uniform"),
            )
        if algorithm == "NaiveBayes":
            return GaussianNB()
        raise ValueError(f"Unsupported classification algorithm '{algorithm}'.")

    if algorithm == "LinearRegression":
        return LinearRegression()
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
            random_state=model_params.get("random_state", 42),
            n_jobs=-1,
        )
    if algorithm == "GradientBoostingRegressor":
        return GradientBoostingRegressor(
            learning_rate=model_params.get("learning_rate", 0.1),
            n_estimators=model_params.get("n_estimators_gb", 100),
            max_depth=model_params.get("max_depth_gb", 3),
            random_state=model_params.get("random_state", 42),
        )
    if algorithm == "SVR":
        return SVR(
            kernel=model_params.get("svr_kernel", "rbf"),
            C=model_params.get("svr_C", 1.0),
        )
    if algorithm == "KNNRegressor":
        return KNeighborsRegressor(
            n_neighbors=model_params.get("knn_neighbors", 5),
            weights=model_params.get("knn_weights", "uniform"),
            n_jobs=-1,
        )
    if algorithm == "Ridge":
        return Ridge(alpha=model_params.get("alpha", 1.0), random_state=model_params.get("random_state", 42))
    if algorithm == "Lasso":
        return Lasso(alpha=model_params.get("alpha", 1.0), random_state=model_params.get("random_state", 42))
    if algorithm == "ElasticNet":
        return ElasticNet(alpha=model_params.get("alpha", 1.0), l1_ratio=model_params.get("l1_ratio", 0.5), random_state=model_params.get("random_state", 42))

    raise ValueError(f"Unsupported regression algorithm '{algorithm}'.")

# =============================================================================
# Hyperparameter Tuning
# =============================================================================

def get_hyperparameter_grid(algorithm: str, task_type: str = "classification") -> Dict[str, List[Any]]:
    """Get hyperparameter search grid for the given algorithm."""
    grids = {
        "LogisticRegression": {
            "logistic_C": [0.01, 0.1, 1.0, 10.0],
            "logistic_solver": ["liblinear", "lbfgs"],
        },
        "DecisionTree": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "RandomForest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "GradientBoosting": {
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5],
        },
        "SVM": {
            "svm_C": [0.1, 1.0, 10.0],
            "svm_kernel": ["linear", "rbf"],
        },
        "KNearestNeighbors": {
            "knn_neighbors": [3, 5, 7],
            "knn_weights": ["uniform", "distance"],
        },
        "RandomForestRegressor": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
        },
        "GradientBoostingRegressor": {
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
        },
        "DecisionTreeRegressor": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "SVR": {
            "svr_kernel": ["linear", "rbf"],
            "svr_C": [0.1, 1.0, 10.0],
        },
        "KNNRegressor": {
            "knn_neighbors": [3, 5, 7],
            "knn_weights": ["uniform", "distance"],
        },
        "Ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "Lasso": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
        "ElasticNet": {"alpha": [0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    }
    return grids.get(algorithm, {})

def _serialize_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    return value

def _append_experiment_record(
    record: Dict[str, Any],
    experiments_dir: str = "experiments",
) -> None:
    ensure_directory(experiments_dir)
    csv_path = os.path.join(experiments_dir, "experiments.csv")
    json_path = os.path.join(experiments_dir, "model_metrics.json")

    record_serialized = _serialize_value(record)
    record_serialized["timestamp"] = str(record_serialized.get("timestamp"))
    record_serialized["metrics"] = json.dumps(record_serialized.get("metrics", {}), default=str)
    record_serialized["hyperparameters"] = json.dumps(record_serialized.get("hyperparameters", {}), default=str)

    df_record = pd.DataFrame([record_serialized])
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        combined = pd.concat([existing_df, df_record], ignore_index=True)
    else:
        combined = df_record
    combined.to_csv(csv_path, index=False)

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                existing_json = json.load(handle)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_json = []
    else:
        existing_json = []

    existing_json.append(record_serialized)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(existing_json, handle, indent=2)

# =============================================================================
# Hyperparameter Tuning
# =============================================================================

def perform_hyperparameter_tuning(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    algorithm: str,
    config: Dict[str, Any],
    task_type: str = "classification",
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
    """
    tune_config = config.get("hyperparameter_tuning", {})
    if not tune_config.get("enabled", False):
        logger.info("Hyperparameter tuning disabled. Using default parameters.")
        return model, {}

    method = tune_config.get("method", "randomized")
    cv_folds = tune_config.get("cv_folds", 3)
    scoring = tune_config.get("scoring", "accuracy" if task_type == "classification" else "r2")
    n_iter = tune_config.get("n_iter", 20)
    n_jobs = tune_config.get("n_jobs", -1)

    param_grid = get_hyperparameter_grid(algorithm, task_type)
    if not param_grid:
        logger.warning("No hyperparameter grid found for '%s'. Skipping tuning.", algorithm)
        return model, {}

    logger.info("Starting hyperparameter tuning using '%s' search with %d CV folds.", method, cv_folds)

    if method == "grid":
        search = GridSearchCV(model, param_grid, cv=cv_folds, scoring=scoring, n_jobs=n_jobs, verbose=0)
    else:
        search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=cv_folds, scoring=scoring, n_jobs=n_jobs, random_state=42, verbose=0)

    search.fit(X_train, y_train)

    logger.info("Best parameters: %s", search.best_params_)
    logger.info("Best CV score: %.4f", search.best_score_)

    return search.best_estimator_, search.best_params_

# =============================================================================
# Cross-Validation
# =============================================================================

def perform_cross_validation(
    model: Any, X: pd.DataFrame, y: pd.Series, config: Dict[str, Any], task_type: str = "classification"
) -> Dict[str, Any]:
    """Perform cross-validation and return scores."""
    cv_config = config.get("cross_validation", {})
    if not cv_config.get("enabled", False):
        return {}

    n_folds = cv_config.get("n_folds", 5)
    scoring = cv_config.get("scoring", "accuracy" if task_type == "classification" else "r2")

    logger.info("Performing %d-fold cross-validation (scoring=%s).", n_folds, scoring)

    scores = cross_val_score(model, X, y, cv=n_folds, scoring=scoring, n_jobs=-1)

    cv_results = {
        "cv_scores": scores.tolist(),
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
        "cv_min": float(scores.min()),
        "cv_max": float(scores.max()),
    }

    logger.info("CV scores: %s", scores)
    logger.info("CV mean: %.4f (±%.4f)", scores.mean(), scores.std())

    return cv_results

# =============================================================================
# MLflow Setup
# =============================================================================

def _normalize_file_tracking_uri(uri: str) -> str:
    if not uri.startswith("file:"):
        return uri

    path_part = uri[5:]
    if not path_part:
        return uri

    path = Path(path_part)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    return path.as_uri()

def _is_valid_file_artifact_location(uri: str) -> bool:
    return uri.startswith("file://")

def setup_mlflow(config: Dict[str, Any]) -> None:
    """Configure MLflow tracking."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    experiment_name = config.get("mlflow_experiment_name", "CARIVIX_AI")
    tracking_uri = config.get("mlflow_tracking_uri", "file:./experiments")
    if tracking_uri.startswith("file:"):
        tracking_uri = _normalize_file_tracking_uri(tracking_uri)
    logger.info("Configuring MLflow tracking URI: %s", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    artifact_location = Path("experiments").resolve().as_uri()
    if experiment is None:
        logger.info("Creating MLflow experiment '%s' with artifact location: %s", experiment_name, artifact_location)
        mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)
    else:
        if not _is_valid_file_artifact_location(experiment.artifact_location):
            fallback_name = f"{experiment_name}_file"
            fallback_experiment = mlflow.get_experiment_by_name(fallback_name)
            if fallback_experiment is None:
                logger.warning(
                    "Existing experiment '%s' has unsupported artifact location '%s'. "
                    "Creating fallback experiment '%s' instead.",
                    experiment_name,
                    experiment.artifact_location,
                    fallback_name,
                )
                mlflow.create_experiment(name=fallback_name, artifact_location=artifact_location)
            else:
                logger.warning(
                    "Using existing fallback experiment '%s' with artifact location: %s",
                    fallback_name,
                    fallback_experiment.artifact_location,
                )
            experiment_name = fallback_name
        else:
            logger.info("Using existing MLflow experiment '%s' with artifact location: %s", experiment_name, experiment.artifact_location)

    mlflow.set_experiment(experiment_name)

# =============================================================================
# SHAP Explainability
# =============================================================================

def compute_shap_explanations(model: Any, X_test: pd.DataFrame, config: Dict[str, Any]) -> Optional[Any]:
    """Compute SHAP explanations for model interpretability."""
    exp_config = config.get("explainability", {})
    if not exp_config.get("enabled", False):
        return None

    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Install with: pip install shap")
        return None

    logger.info("Computing SHAP explanations...")
    background_samples = min(exp_config.get("background_samples", 100), len(X_test))
    explainer = shap.Explainer(model, X_test.sample(background_samples, random_state=42))
    shap_values = explainer(X_test)

    # Save SHAP summary plot
    max_display = exp_config.get("max_display_features", 20)
    fig = plt.figure()
    shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)
    shap_path = os.path.join("experiments", f"shap_summary_{get_timestamp()}.png")
    ensure_directory(os.path.dirname(shap_path))
    plt.savefig(shap_path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved to: %s", shap_path)

    return shap_values

# =============================================================================
# Visualization Helpers
# =============================================================================

def plot_feature_importance(model: Any, feature_names: list, top_n: int = 20) -> str:
    """Generate and save a feature importance plot."""
    if not hasattr(model, "feature_importances_"):
        return ""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    fig, ax = plt.subplots(figsize=(10, max(6, len(top_features) * 0.4)))
    ax.barh(range(len(top_features)), top_importances, align="center", color="steelblue")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {len(top_features)} Feature Importances")
    fig.tight_layout()
    fi_path = os.path.join("experiments", f"feature_importance_{get_timestamp()}.png")
    ensure_directory(os.path.dirname(fi_path))
    plt.savefig(fi_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return fi_path

# =============================================================================
# Main Training Function
# =============================================================================

def train_model(X: pd.DataFrame, y: pd.Series, config: Dict[str, Any], dataset_path: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a machine learning model with MLflow experiment tracking.

    Workflow:
    1. Split data into train/test sets
    2. Optionally perform cross-validation
    3. Optionally perform hyperparameter tuning
    4. Initialize and train the model
    5. Evaluate on test set
    6. Compute SHAP explanations
    7. Log everything to MLflow
    8. Save the trained model
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 60)

    algorithm = config.get("algorithm", "RandomForest")
    task_type = config.get("task_type", "classification")
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)
    model_params = config.get("model_parameters", {})
    model_save_path = config.get("model_save_path", "models/")

    dataset_name = get_dataset_name(dataset_path)
    dataset_version = get_dataset_version(dataset_path)
    timestamp = get_timestamp()

    setup_mlflow(config)

    # Train-Test Split
    logger.info("Splitting data: test_size=%.2f, random_state=%d", test_size, random_state)
    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    logger.info("Train: %s, Test: %s", X_train.shape, X_test.shape)

    run_name = f"{algorithm}_{dataset_name}_{timestamp}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info("MLflow Run started: %s (ID: %s)", run_name, run_id)

        # Log tags
        mlflow.set_tag("dataset_name", dataset_name)
        mlflow.set_tag("dataset_version", dataset_version)
        mlflow.set_tag("model_name", algorithm)
        mlflow.set_tag("task_type", task_type)
        mlflow.set_tag("training_timestamp", timestamp)

        # Log params
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("dataset_name", dataset_name)

        # Initialize model
        model = get_model(algorithm, model_params, task_type)

        # Cross-validation
        cv_results = perform_cross_validation(model, X_train, y_train, config, task_type)
        if cv_results:
            mlflow.log_metric("cv_mean_score", cv_results["cv_mean"])
            mlflow.log_metric("cv_std_score", cv_results["cv_std"])

        # Hyperparameter tuning
        model, best_params = perform_hyperparameter_tuning(model, X_train, y_train, algorithm, config, task_type)
        if best_params:
            for param_name, param_value in best_params.items():
                mlflow.log_param(f"best_{param_name}", param_value)

        # Train
        logger.info("Training %s model...", algorithm)
        training_start = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - training_start
        mlflow.log_metric("training_time_seconds", training_time)
        logger.info("Training completed in %.2f seconds", training_time)

        # Predictions and evaluation
        evaluation_results = evaluate_model(model, X_test, y_test, task_type=task_type)
        metrics = evaluation_results["metrics"]

        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            logger.info("%s: %.4f", metric_name, metric_value)

        # Save and log classification report if available
        if task_type == "classification" and evaluation_results.get("classification_report") is not None:
            report_df = evaluation_results["classification_report"]
            report_path = os.path.join("experiments", f"classification_report_{timestamp}.csv")
            ensure_directory(os.path.dirname(report_path))
            report_df.to_csv(report_path)
            mlflow.log_artifact(report_path)

        # Log evaluation plot artifacts
        for plot_path in evaluation_results.get("plots", []):
            if plot_path:
                mlflow.log_artifact(plot_path)

        # Log feature importance separately for models that support it
        if hasattr(model, "feature_importances_"):
            fi_path = plot_feature_importance(model, X.columns.tolist())
            if fi_path:
                mlflow.log_artifact(fi_path)

        # Save model
        ensure_directory(model_save_path)
        model_filename = f"{algorithm}_{dataset_name}_{timestamp}.pkl"
        model_filepath = os.path.join(model_save_path, model_filename)

        mlflow.sklearn.log_model(sk_model=model, name="model", registered_model_name=f"{algorithm}_{dataset_name}")

        import joblib
        joblib.dump(model, model_filepath)
        logger.info("Model saved to: %s", model_filepath)

        # Log model path
        model_path_note = os.path.join("experiments", f"model_path_{timestamp}.txt")
        with open(model_path_note, "w", encoding="utf-8") as f:
            f.write(model_filepath)
        mlflow.log_artifact(model_path_note)

        # Store experiment record to CSV and JSON
        experiment_record = {
            "timestamp": timestamp,
            "run_id": run_id,
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "algorithm": algorithm,
            "task_type": task_type,
            "training_time_seconds": round(training_time, 4),
            "model_path": model_filepath,
            "hyperparameters": model_params,
            "metrics": metrics,
        }
        _append_experiment_record(experiment_record)

    logger.info("=" * 60)
    logger.info("MODEL TRAINING COMPLETE")
    logger.info("=" * 60)
    return model, metrics

