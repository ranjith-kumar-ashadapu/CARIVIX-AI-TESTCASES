"""
Lightweight NLP module for CARIVIX AI.

Provides a simple analyze(query) function that tries to load a serialized intent
classifier if one is available in the project; otherwise falls back to a
rule-based heuristic for intent routing (RAG / ML / combined).

This file is created from the notebook found in the repository when an
importable nlp module was not available.
"""
from typing import Dict, Any, Optional
import re
import json
import logging
from pathlib import Path

logger = logging.getLogger("CARIVIX_AI.nlp")

_MODEL_PATHS = [
    # Candidate model paths - if you have an intent classifier, add its path here
    "models/intent_classifier.pkl",
    "models/intent_classifier.joblib",
]


def _simple_intent_rules(query: str) -> Dict[str, Any]:
    q = query.lower().strip()

    # Basic heuristics for routing
    ml_keywords = ["predict", "forecast", "estimate", "classify", "probability", "score"]
    rag_keywords = ["what", "who", "when", "where", "how", "explain", "document", "report", "describe"]

    ml_hits = any(tok in q for tok in ml_keywords)
    rag_hits = any(tok in q for tok in rag_keywords)

    if ml_hits and rag_hits:
        route = "combined"
    elif ml_hits:
        route = "ml"
    else:
        # Default to RAG for general Q&A
        route = "rag"

    # Very small intent inference
    intent = "ml_query" if route == "ml" else "rag_query"
    confidence = 0.6 if route in ("ml", "rag") else 0.5

    return {
        "processed_query": q,
        "intent": intent,
        "confidence": confidence,
        "entities": {},
        "route": route,
    }


def analyze(query: str) -> Dict[str, Any]:
    """Return an analysis dict for the given query.

    The returned dict contains at least:
      - processed_query
      - intent
      - confidence
      - entities
      - route   (one of: 'rag', 'ml', 'combined')

    If a serialized classifier is present in one of the candidate model paths,
    an attempt is made to load it and run inference. If loading fails or no
    model is present, the rule-based fallback is used.
    """
    if not query or not str(query).strip():
        return {
            "processed_query": "",
            "intent": "empty",
            "confidence": 0.0,
            "entities": {},
            "route": "rag",
        }

    # NOTE: The repository currently does not include a dedicated intent classifier
    # artifact that the service can reliably load. If you add one, extend
    # _MODEL_PATHS and implement loading logic here.

    # Attempt to load a serialized intent classifier (joblib/pickle). Cache after first load.
    if '_LOADED_INTENT_MODEL' not in globals():
        _LOADED_INTENT_MODEL = None

    if _LOADED_INTENT_MODEL is None:
        try:
            import joblib
            for p in _MODEL_PATHS:
                path = Path(p)
                if path.exists():
                    try:
                        _LOADED_INTENT_MODEL = joblib.load(path)
                        logger.info("Loaded intent classifier from %s", path)
                    except Exception as exc:  # pragma: no cover - optional model loading
                        logger.exception("Failed to load intent model at %s: %s", path, exc)
                    break
        except Exception:
            # joblib not available or other import error — fall back to rules
            _LOADED_INTENT_MODEL = None

    # If a model was successfully loaded, use it to predict
    if _LOADED_INTENT_MODEL is not None:
        try:
            model = _LOADED_INTENT_MODEL
            pred = model.predict([query])
            intent_label = pred[0] if isinstance(pred, (list, tuple)) else pred
            confidence = None
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba([query])[0]
                    # highest probability
                    confidence = float(max(probs))
                except Exception:
                    confidence = None
            # Map label to route if label names differ
            route = "rag" if intent_label == "rag" else ("ml" if intent_label == "ml" else ("combined" if intent_label == "combined" else "rag"))
            return {
                "processed_query": query,
                "intent": str(intent_label),
                "confidence": confidence if confidence is not None else 0.9,
                "entities": {},
                "route": route,
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Intent model prediction failed: %s", exc)

    # Use simple rules as fallback
    return _simple_intent_rules(str(query))


if __name__ == "__main__":
    # Quick manual test
    examples = [
        "What is CARIVIX?",
        "Predict next quarter GDP for dataset X",
        "Explain the methods in the report.docx",
    ]
    for e in examples:
        print(e, "->", analyze(e))
