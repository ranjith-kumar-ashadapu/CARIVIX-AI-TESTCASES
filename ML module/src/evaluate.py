"""
Model Evaluation Module for CARIVIX AI Model Training pipeline.

Provides comprehensive evaluation metrics for both:
- Classification: Accuracy, Precision, Recall, F1 Score, ROC AUC
- Regression: MAE, RMSE, R² Score

Generates visualizations:
- Confusion Matrix (classification)
- Feature Importance Plot
- ROC Curve (binary classification)
- Precision-Recall Curve
- Calibration Curve
- Learning Curves
- SHAP Summary Plot
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve

from src.utils import ensure_directory, get_timestamp

logger = logging.getLogger("CARIVIX_AI")

# =============================================================================
# Classification Metrics
# =============================================================================

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (for ROC AUC).
        average: Averaging method for multi-class metrics.

    Returns:
        Dictionary of metric names and values.
    """
    logger.info("Calculating classification metrics...")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # ROC AUC - only for binary classification
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError as exc:
            logger.warning("Could not compute ROC AUC: %s", exc)
            metrics["roc_auc"] = 0.0
    else:
        logger.debug("ROC AUC not computed (requires binary classification with probabilities).")

    for name, value in metrics.items():
        logger.debug("  %s: %.4f", name, value)

    return metrics

def generate_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate a classification report as a pandas DataFrame.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Names of the classes.

    Returns:
        DataFrame containing the classification report.
    """
    if class_names is None:
        class_names = [str(cls) for cls in sorted(np.unique(y_true))]

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    logger.info("Classification report generated for %d classes.", len(class_names))
    return report_df

# =============================================================================
# Regression Metrics
# =============================================================================

def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary of metric names and values.
    """
    logger.info("Calculating regression metrics...")

    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2_score": r2_score(y_true, y_pred),
    }

    for name, value in metrics.items():
        logger.debug("  %s: %.4f", name, value)

    return metrics

# =============================================================================
# Visualization Functions
# =============================================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> str:
    """
    Plot and save the ROC curve for binary classification.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        File path of the saved plot.
    """
    logger.info("Generating ROC curve plot...")

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        fpr, tpr, color="darkorange", lw=2,
        label=f"ROC curve (AUC = {roc_auc:.4f})"
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # Save plot
    roc_path = os.path.join("experiments", f"roc_curve_{get_timestamp()}.png")
    ensure_directory(os.path.dirname(roc_path))
    plt.savefig(roc_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved to: %s", roc_path)

    return roc_path

def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """
    Plot and save a scatter plot of predicted vs actual values (regression).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        File path of the saved plot.
    """
    logger.info("Generating prediction scatter plot...")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
    ax.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--", lw=2, label="Perfect Prediction"
    )
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Predicted vs Actual Values")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    scatter_path = os.path.join("experiments", f"prediction_scatter_{get_timestamp()}.png")
    ensure_directory(os.path.dirname(scatter_path))
    plt.savefig(scatter_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Prediction scatter plot saved to: %s", scatter_path)

    return scatter_path

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """
    Plot and save residuals distribution (regression).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        File path of the saved plot.
    """
    logger.info("Generating residuals plot...")

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals histogram
    axes[0].hist(residuals, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Residuals")
    axes[0].axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    axes[0].grid(True, alpha=0.3)

    # Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
    axes[1].axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals vs Predicted Values")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    resid_path = os.path.join("experiments", f"residuals_{get_timestamp()}.png")
    ensure_directory(os.path.dirname(resid_path))
    plt.savefig(resid_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Residuals plot saved to: %s", resid_path)

    return resid_path

def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> str:
    """Plot and save a precision-recall curve for binary classification."""
    logger.info("Generating precision-recall curve plot...")

    if len(np.unique(y_true)) != 2:
        logger.warning("Precision-recall curve requires binary labels.")
        return ""

    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="darkgreen", lw=2, label="Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    pr_path = os.path.join("experiments", f"precision_recall_{get_timestamp()}.png")
    ensure_directory(os.path.dirname(pr_path))
    plt.savefig(pr_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Precision-recall curve saved to: %s", pr_path)

    return pr_path

def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    bins: int = 10,
) -> str:
    """Plot and save a calibration curve for probabilistic predictions."""
    logger.info("Generating calibration curve plot...")

    if len(np.unique(y_true)) != 2:
        logger.warning("Calibration curve requires binary labels.")
        return ""

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=bins
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mean_predicted_value, fraction_of_positives, marker="o", linewidth=2, label="Observed")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    calib_path = os.path.join("experiments", f"calibration_curve_{get_timestamp()}.png")
    ensure_directory(os.path.dirname(calib_path))
    plt.savefig(calib_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Calibration curve saved to: %s", calib_path)

    return calib_path

def plot_learning_curve(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "classification",
    cv: int = 3,
    train_sizes: Optional[np.ndarray] = None,
) -> str:
    """Plot and save a learning curve for the model."""
    logger.info("Generating learning curve plot...")

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)

    scoring = "accuracy" if task_type == "classification" else "neg_mean_squared_error"
    try:
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            train_sizes=train_sizes,
            n_jobs=-1,
        )
    except Exception as exc:
        logger.warning("Learning curve generation failed: %s", exc)
        return ""

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes_abs, train_mean, label="Training Score", color="tab:blue")
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.plot(train_sizes_abs, test_mean, label="Cross-Validation Score", color="tab:orange")
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.2)
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Score")
    ax.set_title("Learning Curve")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    learning_path = os.path.join("experiments", f"learning_curve_{get_timestamp()}.png")
    ensure_directory(os.path.dirname(learning_path))
    plt.savefig(learning_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Learning curve saved to: %s", learning_path)

    return learning_path

def plot_shap_summary(model: Any, X: pd.DataFrame) -> str:
    """Plot and save a SHAP summary plot when SHAP is available."""
    logger.info("Generating SHAP summary plot...")

    try:
        import shap
    except ImportError:
        logger.warning("SHAP is not installed; skipping SHAP plot.")
        return ""

    try:
        explainer = shap.Explainer(model, X.sample(min(100, len(X)), random_state=42))
        shap_values = explainer(X)
    except Exception as exc:
        logger.warning("Unable to compute SHAP values: %s", exc)
        return ""

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    shap_path = os.path.join("experiments", f"shap_summary_{get_timestamp()}.png")
    ensure_directory(os.path.dirname(shap_path))
    plt.savefig(shap_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("SHAP summary plot saved to: %s", shap_path)

    return shap_path

# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str = "classification",
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test target.
        task_type: Type of task ('classification' or 'regression').

    Returns:
        Dictionary containing:
        - 'metrics': Performance metrics
        - 'predictions': Model predictions
        - 'probabilities': Predicted probabilities (classification only)
        - 'classification_report': Classification report DataFrame
        - 'plots': List of generated plot file paths
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL EVALUATION")
    logger.info("=" * 60)

    results: Dict[str, Any] = {
        "metrics": {},
        "predictions": None,
        "probabilities": None,
        "classification_report": None,
        "plots": [],
    }

    # --- Generate Predictions ---
    logger.info("Generating predictions on test set...")
    y_pred = model.predict(X_test)
    results["predictions"] = y_pred

    if task_type == "classification":
        # --- Classification Evaluation ---
        logger.info("Evaluating classification model...")

        # Get predicted probabilities
        try:
            y_prob = model.predict_proba(X_test)
            is_binary = y_prob.shape[1] == 2
            results["probabilities"] = y_prob[:, 1] if is_binary else y_prob
        except (AttributeError, NotImplementedError):
            y_prob = None
            results["probabilities"] = None

        # Calculate metrics
        metrics = calculate_classification_metrics(y_test, y_pred, results["probabilities"])
        results["metrics"] = metrics

        # Generate classification report
        class_names = [str(cls) for cls in sorted(y_test.unique())]
        report_df = generate_classification_report_df(y_test, y_pred, class_names)
        results["classification_report"] = report_df

        # Generate plots
        cm_path = plot_confusion_matrix(y_test, y_pred, class_names)
        if cm_path:
            results["plots"].append(cm_path)

        # ROC curve for binary classification
        if is_binary and results["probabilities"] is not None:
            roc_path = plot_roc_curve(y_test, results["probabilities"])
            if roc_path:
                results["plots"].append(roc_path)

            pr_path = plot_precision_recall_curve(y_test, results["probabilities"])
            if pr_path:
                results["plots"].append(pr_path)

            calib_path = plot_calibration_curve(y_test, results["probabilities"])
            if calib_path:
                results["plots"].append(calib_path)

        learning_path = plot_learning_curve(model, X_test, y_test, task_type=task_type)
        if learning_path:
            results["plots"].append(learning_path)

        shap_path = plot_shap_summary(model, X_test)
        if shap_path:
            results["plots"].append(shap_path)

    elif task_type == "regression":
        # --- Regression Evaluation ---
        logger.info("Evaluating regression model...")

        metrics = calculate_regression_metrics(y_test, y_pred)
        results["metrics"] = metrics

        # Generate plots
        scatter_path = plot_prediction_scatter(y_test, y_pred)
        if scatter_path:
            results["plots"].append(scatter_path)

        resid_path = plot_residuals(y_test, y_pred)
        if resid_path:
            results["plots"].append(resid_path)

        learning_path = plot_learning_curve(model, X_test, y_test, task_type=task_type)
        if learning_path:
            results["plots"].append(learning_path)

        shap_path = plot_shap_summary(model, X_test)
        if shap_path:
            results["plots"].append(shap_path)

    else:
        raise ValueError(
            f"Invalid task_type '{task_type}'. Choose 'classification' or 'regression'."
        )

    logger.info("=" * 60)
    logger.info("MODEL EVALUATION COMPLETE")
    logger.info("=" * 60)

    return results

# =============================================================================
# Re-export confusion matrix plot from train.py to maintain consistency
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
) -> str:
    """
    Generate and save a confusion matrix plot.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Names of classes (optional).

    Returns:
        File path of the saved plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()

    cm_path = os.path.join("experiments", f"confusion_matrix_{get_timestamp()}.png")
    ensure_directory(os.path.dirname(cm_path))
    plt.savefig(cm_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to: %s", cm_path)

    return cm_path


