"""
Prediction Module for CARIVIX AI Model Training pipeline.

Provides:
- Loading trained models
- Making predictions on new data
- Batch prediction on datasets
- Probability prediction (classification)
"""

import copy
import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from src.feature_engineering import run_feature_engineering_pipeline
from src.preprocess import run_preprocessing_pipeline
from src.utils import setup_logger, load_config, load_dataframe, ensure_directory

logger = logging.getLogger("CARIVIX_AI")

def _align_features(X: pd.DataFrame, model: Any) -> pd.DataFrame:
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        X_aligned = X.reindex(columns=expected_features, fill_value=0)
        return X_aligned
    return X

def _prepare_prediction_data(
    data_path: str,
    config: Dict[str, Any],
    model: Any,
) -> pd.DataFrame:
    df = load_dataframe(data_path)
    target_column = config.get("target_column", "target")
    config_copy = copy.deepcopy(config)
    preprocess_cfg = config_copy.setdefault("preprocessing", {})
    preprocess_cfg["handle_imbalanced"] = False

    target_in_data = target_column in df.columns
    preprocess_target = target_column if target_in_data else None

    X, y, _ = run_preprocessing_pipeline(df, config_copy, preprocess_target)
    X_engineered, _ = run_feature_engineering_pipeline(X, y, config_copy)
    X_aligned = _align_features(X_engineered, model)
    return X_aligned

# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_path: str) -> Any:
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the saved model file (.pkl or .joblib).

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
        ValueError: If the model file format is not supported.
    """
    logger.info("Loading model from: %s", model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Determine file extension
    _, ext = os.path.splitext(model_path)

    try:
        if ext in [".pkl", ".joblib"]:
            model = joblib.load(model_path)
        else:
            raise ValueError(
                f"Unsupported model format '{ext}'. "
                f"Supported formats: .pkl, .joblib"
            )
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        raise

    logger.info("Model loaded successfully: %s", type(model).__name__)
    return model

def load_latest_model(models_dir: str = "models/") -> Tuple[Any, str]:
    """
    Load the latest trained model from a directory.

    Args:
        models_dir: Directory containing saved models.

    Returns:
        Tuple of (loaded model, path to the model file).

    Raises:
        FileNotFoundError: If no model files are found.
    """
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # Find all model files
    model_files = [
        f for f in os.listdir(models_dir)
        if f.endswith((".pkl", ".joblib"))
    ]

    if not model_files:
        raise FileNotFoundError(
            f"No model files found in '{models_dir}'."
        )

    # Get the most recent model file
    latest_model = max(
        model_files,
        key=lambda f: os.path.getmtime(os.path.join(models_dir, f)),
    )
    latest_path = os.path.join(models_dir, latest_model)

    logger.info("Latest model found: %s", latest_model)
    return load_model(latest_path), latest_path

# =============================================================================
# Prediction Functions
# =============================================================================

def predict(
    model: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Make predictions using a trained model.

    Args:
        model: Trained model object.
        X: Feature DataFrame.

    Returns:
        Array of predictions.
    """
    logger.info("Making predictions on data with shape: %s", X.shape)

    try:
        predictions = model.predict(X)
        logger.info("Predictions generated successfully.")
        return predictions
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise

def predict_proba(
    model: Any,
    X: pd.DataFrame,
) -> Optional[np.ndarray]:
    """
    Get prediction probabilities (for classification models).

    Args:
        model: Trained classification model.
        X: Feature DataFrame.

    Returns:
        Array of predicted probabilities, or None if the model doesn't support it.
    """
    logger.info("Getting prediction probabilities for data with shape: %s", X.shape)

    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            logger.info(
                "Prediction probabilities generated. Shape: %s", probabilities.shape
            )
            return probabilities
        else:
            logger.warning("Model does not support predict_proba.")
            return None
    except Exception as exc:
        logger.error("Probability prediction failed: %s", exc)
        raise

# =============================================================================
# Advanced Prediction Utilities
# =============================================================================

def get_feature_importance(model: Any, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """Return a DataFrame of feature importance values when available."""
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        return importance_df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    if hasattr(model, "coef_"):
        coef_values = np.asarray(model.coef_).ravel()
        importance_df = pd.DataFrame({"feature": feature_names, "importance": np.abs(coef_values)})
        return importance_df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    return pd.DataFrame(columns=["feature", "importance"])

def predict_with_intervals(
    model: Any,
    X: pd.DataFrame,
    residual_scale: Optional[float] = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Generate point predictions with approximate prediction intervals."""
    predictions = predict(model, X)
    if residual_scale is None:
        residual_scale = 0.0

    z_value = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.0
    half_width = z_value * float(residual_scale)
    results = pd.DataFrame({
        "prediction": predictions,
        "lower_bound": predictions - half_width,
        "upper_bound": predictions + half_width,
    })
    return results

def compare_models(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y_true: Optional[pd.Series] = None,
    task_type: str = "classification",
) -> pd.DataFrame:
    """Compare multiple loaded models on the same feature matrix."""
    results = []
    for name, model in models.items():
        preds = model.predict(X)
        if task_type == "classification" and y_true is not None:
            metrics = {
                "model": name,
                "accuracy": accuracy_score(y_true, preds),
            }
        elif task_type == "regression" and y_true is not None:
            metrics = {
                "model": name,
                "mse": mean_squared_error(y_true, preds),
                "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
                "r2_score": r2_score(y_true, preds),
            }
        else:
            metrics = {"model": name}
        results.append(metrics)

    return pd.DataFrame(results)

# =============================================================================
# Batch Prediction
# =============================================================================

def batch_predict(
    model: Any,
    data_path: str,
    output_path: Optional[str] = None,
    include_probabilities: bool = False,
) -> pd.DataFrame:
    """
    Run batch predictions on a dataset and optionally save results.

    Args:
        model: Trained model object.
        data_path: Path to the input data CSV file.
        output_path: Path to save predictions (optional).
        include_probabilities: Whether to include predicted probabilities.

    Returns:
        DataFrame containing predictions and optionally probabilities.
    """
    logger.info("Starting batch prediction on: %s", data_path)

    # Load data
    df = load_dataframe(data_path)

    # Make predictions
    predictions = predict(model, df)
    results_df = df.copy()
    results_df["prediction"] = predictions

    # Include probabilities if requested
    if include_probabilities:
        probabilities = predict_proba(model, df)
        if probabilities is not None:
            n_classes = probabilities.shape[1]
            for i in range(n_classes):
                results_df[f"probability_class_{i}"] = probabilities[:, i]

    # Save results
    if output_path:
        ensure_directory(os.path.dirname(output_path))
        results_df.to_csv(output_path, index=False)
        logger.info("Batch predictions saved to: %s", output_path)

    logger.info("Batch prediction complete. Results shape: %s", results_df.shape)
    return results_df

# =============================================================================
# Single Prediction
# =============================================================================

def predict_single(
    model: Any,
    features: Dict[str, Any],
    feature_names: List[str],
) -> Union[int, float, str]:
    """
    Make a prediction on a single sample.

    Args:
        model: Trained model object.
        features: Dictionary of feature name to value.
        feature_names: List of feature names in the order expected by the model.

    Returns:
        Predicted value for the single sample.
    """
    logger.info("Making single prediction...")

    # Create DataFrame from single sample
    sample_data = []
    for name in feature_names:
        sample_data.append(features.get(name, 0.0))

    sample_df = pd.DataFrame([sample_data], columns=feature_names)
    logger.debug("Single sample shape: %s", sample_df.shape)

    # Predict
    prediction = predict(model, sample_df)
    logger.info("Single prediction: %s", prediction[0])

    return prediction[0]

# =============================================================================
# Main Prediction Pipeline
# =============================================================================

def run_prediction_pipeline(
    model_path: str,
    data_path: str,
    output_path: Optional[str] = None,
    include_proba: bool = False,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the complete prediction pipeline: load model, load data, preprocess features, predict, and save.

    Args:
        model_path: Path to the trained model file.
        data_path: Path to the input data CSV file.
        output_path: Path to save predictions (optional).
        include_proba: Whether to include predicted probabilities.
        config_path: Optional path to YAML config for preprocessing.

    Returns:
        DataFrame with predictions.
    """
    logger.info("=" * 60)
    logger.info("STARTING PREDICTION PIPELINE")
    logger.info("=" * 60)

    # Load model
    model = load_model(model_path)

    # Prepare input features
    if config_path:
        config = load_config(config_path)
        X = _prepare_prediction_data(data_path, config, model)
        df = load_dataframe(data_path)
    else:
        df = load_dataframe(data_path)
        X = df.copy()

    # Make predictions
    predictions = predict(model, X)
    results_df = df.loc[X.index].copy()
    results_df["prediction"] = predictions

    # Include probabilities
    if include_proba:
        probabilities = predict_proba(model, X)
        if probabilities is not None:
            n_classes = probabilities.shape[1]
            for i in range(n_classes):
                results_df[f"probability_class_{i}"] = probabilities[:, i]

    # Save results
    if output_path:
        ensure_directory(os.path.dirname(output_path))
        results_df.to_csv(output_path, index=False)
        logger.info("Predictions saved to: %s", output_path)

    logger.info("=" * 60)
    logger.info("PREDICTION PIPELINE COMPLETE")
    logger.info("=" * 60)

    return results_df


