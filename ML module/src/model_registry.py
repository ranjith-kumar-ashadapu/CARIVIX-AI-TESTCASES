"""
Model Registry Module for CARIVIX AI Model Training pipeline.

Provides centralized model saving/loading:
- Save trained models to the models/ directory using joblib
- Load models by name or path
- List all registered models
- Track model metadata (name, timestamp, path, metrics)
"""

import os
import glob
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

from src.utils import ensure_directory, get_timestamp

logger = logging.getLogger("CARIVIX_AI")

# =============================================================================
# Constants
# =============================================================================

MODELS_DIR = "models"

# =============================================================================
# Model Saving
# =============================================================================

def save_model(
    model: Any,
    model_name: str,
    dataset_name: str,
    models_dir: str = MODELS_DIR,
    timestamp: Optional[str] = None,
) -> str:
    """
    Save a trained model to disk using joblib.

    Args:
        model: Trained model object.
        model_name: Name of the model/algorithm (e.g., 'RandomForest', 'LogisticRegression').
        dataset_name: Name of the dataset used for training.
        models_dir: Directory to save models.
        timestamp: Optional timestamp string. If None, auto-generated.

    Returns:
        Path to the saved model file.

    Raises:
        Exception: If model saving fails.
    """
    if timestamp is None:
        timestamp = get_timestamp()

    ensure_directory(models_dir)

    # Sanitize names for file system
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    safe_dataset_name = dataset_name.replace("/", "_").replace("\\", "_")

    filename = f"{safe_model_name}_{safe_dataset_name}_{timestamp}.pkl"
    filepath = os.path.join(models_dir, filename)

    try:
        joblib.dump(model, filepath)
        logger.info("Model saved to: %s", filepath)
    except Exception as exc:
        logger.error("Failed to save model to %s: %s", filepath, exc)
        raise

    return filepath

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
        ValueError: If the file format is not supported.
    """
    logger.info("Loading model from: %s", model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    _, ext = os.path.splitext(model_path)
    if ext not in (".pkl", ".joblib"):
        raise ValueError(
            f"Unsupported model format '{ext}'. Supported formats: .pkl, .joblib"
        )

    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully: %s", type(model).__name__)
        return model
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        raise

# =============================================================================
# Model Listing & Discovery
# =============================================================================

def list_models(
    models_dir: str = MODELS_DIR,
    pattern: str = "*.pkl",
) -> List[Dict[str, Any]]:
    """
    List all saved models in the models directory.

    Args:
        models_dir: Directory containing models.
        pattern: File glob pattern to match model files.

    Returns:
        List of dicts with model info: filename, path, size, modified time.
    """
    if not os.path.exists(models_dir):
        logger.warning("Models directory not found: %s", models_dir)
        return []

    model_files = glob.glob(os.path.join(models_dir, pattern))
    model_files.sort(key=os.path.getmtime, reverse=True)

    models_info = []
    for filepath in model_files:
        stat = os.stat(filepath)
        info = {
            "filename": os.path.basename(filepath),
            "path": filepath,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
        # Parse model name from filename (format: ModelName_Dataset_Timestamp.pkl)
        name_parts = os.path.splitext(os.path.basename(filepath))[0].split("_")
        if len(name_parts) >= 3:
            info["model_name"] = name_parts[0]
            info["dataset_name"] = name_parts[1]
        else:
            info["model_name"] = name_parts[0] if name_parts else "Unknown"
            info["dataset_name"] = "Unknown"

        models_info.append(info)

    logger.info("Found %d models in %s", len(models_info), models_dir)
    return models_info

def find_latest_model(
    model_name: str,
    models_dir: str = MODELS_DIR,
) -> Optional[str]:
    """
    Find the latest saved model file for a given model name.

    Args:
        model_name: Name of the model to find.
        models_dir: Directory containing models.

    Returns:
        Path to the latest model file, or None if not found.
    """
    models = list_models(models_dir)
    matching = [m for m in models if m.get("model_name") == model_name]

    if not matching:
        logger.warning("No models found matching '%s' in %s", model_name, models_dir)
        return None

    latest = matching[0]  # Already sorted by modification time descending
    logger.info("Latest model for '%s': %s", model_name, latest["path"])
    return latest["path"]

def get_model_summary(models_dir: str = MODELS_DIR) -> pd.DataFrame:
    """
    Get a summary DataFrame of all saved models.

    Args:
        models_dir: Directory containing models.

    Returns:
        DataFrame with model summary information.
    """
    models = list_models(models_dir)
    if not models:
        return pd.DataFrame(columns=["model_name", "dataset_name", "size_mb", "modified", "path"])

    df = pd.DataFrame(models)
    logger.info("Model summary generated with %d models.", len(df))
    return df

# =============================================================================
# Model Deletion
# =============================================================================

def delete_model(model_path: str) -> bool:
    """
    Delete a saved model file.

    Args:
        model_path: Path to the model file to delete.

    Returns:
        True if deletion was successful, False otherwise.
    """
    try:
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info("Deleted model: %s", model_path)
            return True
        logger.warning("Model file not found for deletion: %s", model_path)
        return False
    except Exception as exc:
        logger.error("Failed to delete model %s: %s", model_path, exc)
        return False

