import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import run_feature_engineering_pipeline
from src.preprocess import run_preprocessing_pipeline
from src.utils import load_config

logger = logging.getLogger("CARIVIX_AI")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

_REQUIRED_SYSTEM_FIELDS = [
    "age",
    "income",
    "credit_score",
    "loan_amount",
    "years_employed",
    "education",
    "employment_status",
    "marital_status",
    "housing_type",
    "application_date",
]


class ModelService:
    """Load and serve trained ML models for inference with project preprocessing."""

    def __init__(
        self,
        models_dir: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self.project_root = PROJECT_ROOT
        self.config_path = str(config_path or DEFAULT_CONFIG_PATH)
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        # Track whether the caller explicitly provided a models_dir. If so, do not fall back to
        # discovering model paths from experiments/experiments.csv. Tests rely on an explicitly-set
        # empty models_dir to mean "no models available".
        self._models_dir_explicit = models_dir is not None
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.load_errors: List[Dict[str, str]] = []
        self.default_model_name: Optional[str] = None
        self.default_model_path: Optional[str] = None
        self.config = self._load_config()
        self._discover_and_load_models()

    def _load_config(self) -> Dict[str, Any]:
        config_path = Path(self.config_path)
        if not config_path.exists():
            logger.warning("Config file not found at %s; using empty config.", config_path)
            return {}
        return load_config(str(config_path), base_dir=str(self.project_root))

    @staticmethod
    def _derive_model_name(model_path: str) -> str:
        model_stem = Path(model_path).stem
        for marker in ("_processed_", "_dataset_", "_"):
            if marker in model_stem:
                return model_stem.split(marker)[0]
        return model_stem

    @staticmethod
    def _resolve_model_type(model: Any) -> str:
        if hasattr(model, "classes_") or hasattr(model, "predict_proba"):
            return "classification"
        return "regression"

    def _discover_and_load_models(self) -> None:
        """Discover model files and attempt to load them.

        This implementation is more flexible than the previous one:
        - searches recursively for common model file extensions (.pkl, .joblib, .sav, .model)
        - if no files are found, will look for model paths referenced in experiments/experiments.csv
        - records load errors for diagnostics
        """
        if not self.models_dir.exists():
            logger.warning("No models directory found at %s", self.models_dir)
            return

        # Collect candidate files from models directory (recursive) for multiple extensions
        candidate_files = []
        for pattern in ("*.pkl", "*.joblib", "*.sav", "*.model"):
            candidate_files.extend(list(self.models_dir.rglob(pattern)))

        # Sort by modification time (newest first) and deduplicate
        candidate_files = sorted(set(candidate_files), key=lambda p: p.stat().st_mtime, reverse=True)

        # If nothing found in models dir, and the models_dir was not explicitly provided,
        # try to resolve model paths referenced in experiments CSV. When a caller explicitly
        # sets models_dir (even if empty), do not consult experiments.csv — tests rely on this
        # behavior to simulate "no models available".
        if not candidate_files and not getattr(self, "_models_dir_explicit", False):
            experiments_csv = self.project_root / "experiments" / "experiments.csv"
            if experiments_csv.exists():
                try:
                    import csv

                    with experiments_csv.open("r", encoding="utf-8") as fh:
                        reader = csv.DictReader(fh)
                        for row in reader:
                            mp = row.get("model_path")
                            if not mp:
                                continue
                            candidate = (self.project_root / mp).resolve()
                            if candidate.exists():
                                candidate_files.append(candidate)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Failed to read experiments CSV for model discovery: %s", exc)

        for model_path in candidate_files:
            try:
                model_name = self._derive_model_name(str(model_path))
                model = joblib.load(str(model_path))
            except Exception as exc:  # pragma: no cover - defensive, logged internally only
                # Record a concise error message and provide an operator hint for common causes
                exc_msg = str(exc)
                hint = None
                # Common pickle incompatibility: NumPy BitGenerator/RNG internals changed
                if "BitGenerator" in exc_msg or "is not a known BitGenerator" in exc_msg or "MT19937" in exc_msg:
                    hint = "Incompatible NumPy / RNG internals when unpickling. Align NumPy version with the training environment (e.g., try numpy >=2.4) or recreate the model with a compatible serialization."
                # Missing dependency while unpickling (e.g., xgboost not installed)
                elif "No module named" in exc_msg:
                    # extract module name for hint
                    try:
                        missing = exc_msg.split("No module named")[1].strip().strip("'\"")
                        hint = f"Missing dependency during model load: {missing}. Install the missing package in the runtime environment."
                    except Exception:
                        hint = "Missing dependency required to unpickle the model. Install the training-time packages in the runtime environment."
                err_info = {"path": str(model_path.name), "error": exc_msg}
                if hint:
                    err_info["hint"] = hint
                self.load_errors.append(err_info)
                # Log full exception for operators (internal logs only) and include the hint in the log
                if hint:
                    logger.exception("Failed to load model %s: %s; hint: %s", model_path.name, exc, hint)
                else:
                    logger.exception("Failed to load model %s: %s", model_path.name, exc)
                continue

            # Guarantee unique names for models with the same derived name
            base_name = model_name
            i = 1
            while model_name in self.loaded_models:
                model_name = f"{base_name}_{i}"
                i += 1

            self.loaded_models[model_name] = {
                "model": model,
                "path": str(model_path),
                "algorithm": type(model).__name__,
                "type": self._resolve_model_type(model),
                "status": "loaded",
            }

            if self.default_model_name is None:
                self.default_model_name = model_name
                self.default_model_path = str(model_path)

    def reload_models(self) -> None:
        """Clear any previously-loaded models and re-run discovery.

        Call this when models were added to disk after the service started.
        """
        self.loaded_models = {}
        self.load_errors = []
        self.default_model_name = None
        self.default_model_path = None
        self._discover_and_load_models()

    def has_loaded_models(self) -> bool:
        return bool(self.loaded_models)

    def list_models(self) -> List[Dict[str, Any]]:
        model_entries: List[Dict[str, Any]] = []
        for name, metadata in self.loaded_models.items():
            model_entries.append(
                {
                    "name": name,
                    "type": metadata.get("type", self.config.get("task_type", "classification")),
                    "status": metadata.get("status", "loaded"),
                }
            )
        return model_entries

    def get_load_errors(self) -> List[Dict[str, str]]:
        """Return any errors encountered while loading model files during service initialization."""
        return list(self.load_errors)

    def get_model(self, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not self.loaded_models:
            return None

        selected_name = model_name or self.default_model_name
        if selected_name is None:
            selected_name = next(iter(self.loaded_models))
        if selected_name not in self.loaded_models:
            return None
        return self.loaded_models[selected_name]

    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        model_record = self.get_model(model_name)
        if model_record is None:
            raise RuntimeError("No trained model is currently available.")

        resolved_name = model_name or self.default_model_name or next(iter(self.loaded_models))
        model = model_record["model"]
        task_type = self.config.get("task_type") or model_record.get("type") or "classification"
        return {
            "name": resolved_name,
            "type": model_record.get("type", task_type),
            "task_type": task_type,
            "algorithm": model_record.get("algorithm", type(model).__name__),
            "status": model_record.get("status", "loaded"),
            "supported_prediction_mode": "classification" if model_record.get("type") == "classification" else task_type,
        }

    @staticmethod
    def _normalize_input_row(raw_input: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(raw_input)

        for key in ("age", "income", "credit_score", "loan_amount", "years_employed"):
            if key in normalized and normalized[key] is not None:
                normalized[key] = float(normalized[key])

        if "application_date" in normalized and normalized["application_date"] is not None:
            normalized["application_date"] = str(normalized["application_date"]).strip()

        return normalized

    def _prepare_features_for_model(self, raw_input: Dict[str, Any], model: Any) -> pd.DataFrame:
        if not isinstance(raw_input, dict):
            raise ValueError("Input data must be a dictionary of feature values.")

        missing_fields = [
            field for field in _REQUIRED_SYSTEM_FIELDS if field not in raw_input or raw_input.get(field) is None
        ]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        payload = self._normalize_input_row(raw_input)
        dataset_rel_path = self.config.get("dataset_path", "data/raw/dataset.csv")
        reference_dataset_path = (self.project_root / dataset_rel_path).resolve()
        if not reference_dataset_path.exists():
            raise FileNotFoundError(
                "Reference training dataset not found. Inference cannot be prepared with the required feature schema."
            )

        reference_df = pd.read_csv(reference_dataset_path)
        augmented_df = pd.concat([reference_df, pd.DataFrame([payload])], ignore_index=True)
        augmented_df.loc[augmented_df.index[-1], "target"] = 0

        X, y, _ = run_preprocessing_pipeline(augmented_df, self.config, target_column="target")
        X_engineered, _ = run_feature_engineering_pipeline(X, y, self.config)
        single_row = X_engineered.iloc[[-1]].copy()

        if not hasattr(model, "feature_names_in_"):
            raise RuntimeError("The loaded model does not expose the expected feature metadata.")

        expected_features = list(model.feature_names_in_)
        aligned_features = single_row.reindex(columns=expected_features, fill_value=0.0)
        missing_features = [feature for feature in expected_features if feature not in single_row.columns]
        if missing_features:
            raise ValueError(
                "Input data cannot be transformed into the model feature set. Missing model features: "
                + ", ".join(missing_features[:10])
            )
        return aligned_features

    def predict(self, input_data: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
        model_record = self.get_model(model_name)
        if model_record is None:
            raise RuntimeError("No trained model is currently available.")

        model = model_record["model"]
        try:
            X = self._prepare_features_for_model(input_data, model)
            prediction = model.predict(X)
            prediction_value = prediction[0]
            resolved_prediction = int(prediction_value) if isinstance(prediction_value, (np.integer, int)) else float(prediction_value)

            response: Dict[str, Any] = {
                "success": True,
                "model": model_name or self.default_model_name or next(iter(self.loaded_models)),
                "prediction": resolved_prediction,
            }

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)[0]
                confidence = float(np.max(probabilities))
                response["confidence"] = confidence
                if hasattr(model, "classes_"):
                    class_labels = [int(label) for label in getattr(model, "classes_", [])]
                    probability_map = {str(label): float(prob) for label, prob in zip(class_labels, probabilities)}
                    response["probability"] = float(probability_map.get(str(resolved_prediction), confidence))
            return response
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover - logged internally only
            logger.exception("Prediction failed for input: %s", input_data)
            raise RuntimeError("Model inference failed") from exc

    def predict_batch(self, records: List[Dict[str, Any]], model_name: Optional[str] = None) -> Dict[str, Any]:
        if not records:
            raise ValueError("At least one record is required for batch prediction.")

        selected_model = model_name or self.default_model_name
        predictions = []
        for record in records:
            predictions.append(self.predict(record, model_name=selected_model)["prediction"])

        return {
            "success": True,
            "model": selected_model or next(iter(self.loaded_models)),
            "predictions": predictions,
            "count": len(predictions),
        }


service = ModelService()

__all__ = ["ModelService", "service"]
