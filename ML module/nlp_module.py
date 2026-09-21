"""
Enterprise NLP Intelligence & Routing Module for CARIVIX AI.
=============================================================

Capabilities:
  1. Intent Classification:
     - Model-driven inference via models/intent_classifier.joblib (TF-IDF + LogisticRegression)
     - Supported intents: GIS_VIEW, PREDICTION, DATA_METRIC, FAQ, UNKNOWN_INTENT
     - Fallback rule-based heuristic routing
  2. Domain Entity Extraction:
     - Geographic locations (States, Districts)
     - Economic metrics & indicators (GDP, rainfall, revenue, default risk, inflation)
     - Temporal tokens (Years, Dates)
     - Structured loan & demographic features (age, income, credit score, loan amount, employment)
  3. Structured Query Generation:
     - Deterministic transformation of natural language into downstream API contracts (Backend, ML, WebGIS)
     - SLA < 200 ms execution
  4. Speech-to-Text (STT) Quality & Contract Utilities:
     - Levenshtein Word Error Rate (WER) computation for voice pipeline verification
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("CARIVIX_AI.nlp")

HERE = Path(__file__).resolve().parent
_MODEL_PATHS = [
    HERE / "models" / "intent_classifier.joblib",
    HERE / "models" / "intent_classifier.pkl",
]

_LOADED_INTENT_MODEL = None

# Known administrative entities for Telangana and India
KNOWN_LOCATIONS = [
    "telangana", "andhra pradesh", "karnataka", "maharashtra", "tamil nadu",
    "adilabad", "bhadradri kothagudem", "hyderabad", "jagtial", "jangaon",
    "jayashankar bhupalpally", "jogulamba gadwal", "kamareddy", "karimnagar",
    "khammam", "kumuram bheem asifabad", "mahabubabad", "mahabubnagar",
    "mancherial", "medak", "medchal malkajgiri", "mulugu", "nagarkurnool",
    "nalgonda", "narayanpet", "nirmal", "nizamabad", "peddapalli",
    "rajanna sircilla", "ranga reddy", "rangareddy", "sangareddy", "siddipet",
    "suryapet", "vikarabad", "wanaparthy", "warangal", "yadadri bhuvanagiri"
]

KNOWN_METRICS = [
    "rainfall", "precipitation", "gdp", "revenue", "default", "credit risk",
    "churn", "inflation", "unemployment", "density", "density index",
    "growth rate", "population", "loan amount", "credit score", "sales"
]


def _get_intent_model():
    global _LOADED_INTENT_MODEL
    if _LOADED_INTENT_MODEL is None:
        try:
            import joblib
            for path in _MODEL_PATHS:
                if path.exists():
                    _LOADED_INTENT_MODEL = joblib.load(path)
                    logger.info("Successfully loaded intent classifier from %s", path)
                    break
        except Exception as exc:
            logger.warning("Failed to load intent classifier: %s", exc)
            _LOADED_INTENT_MODEL = None
    return _LOADED_INTENT_MODEL


def extract_entities(query: str) -> Dict[str, Any]:
    """Extract domain entities (locations, metrics, dates, loan features) from natural language."""
    q_clean = str(query).strip()
    q_lower = q_clean.lower()

    # 1. Locations
    locations: List[str] = []
    for loc in KNOWN_LOCATIONS:
        # Match whole words or boundary patterns
        pattern = r"\b" + re.escape(loc) + r"\b"
        if re.search(pattern, q_lower):
            # Capitalize each word for canonical representation
            locations.append(" ".join(w.capitalize() for w in loc.split()))

    # 2. Metrics
    metrics: List[str] = []
    for met in KNOWN_METRICS:
        pattern = r"\b" + re.escape(met) + r"\b"
        if re.search(pattern, q_lower):
            metrics.append(met)

    # 3. Dates & Years
    dates: List[str] = []
    # Match YYYY-MM-DD
    for d in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", q_clean):
        dates.append(d)
    # Match standalone 4-digit years (2010 - 2035)
    for y in re.findall(r"\b(20[1-3][0-9])\b", q_clean):
        if y not in dates:
            dates.append(y)

    # 4. Numerical Loan/Demographic features (for ML inference queries)
    features: Dict[str, Any] = {}
    
    # Age: e.g. "35-year-old", "age 35", "35 yo"
    age_match = re.search(r"(\d+)\s*(?:-| )(?:year|yr)(?:s)?(?:-| )old|\bage\s*(?:of|:)?\s*(\d+)", q_lower)
    if age_match:
        features["age"] = float(age_match.group(1) or age_match.group(2))

    # Income: e.g. "income $50,000", "income 50000", "earning $30,000"
    income_match = re.search(r"(?:income|earning|salary)(?:\s+of)?(?:\s+is)?(?:\s*:)?\s*\$?([\d,]+)", q_lower)
    if income_match:
        features["income"] = float(income_match.group(1).replace(",", ""))

    # Credit Score: e.g. "credit score 680", "score of 750"
    cs_match = re.search(r"(?:credit score|score)(?:\s+of)?(?:\s*:)?\s*(\d{3})\b", q_lower)
    if cs_match:
        features["credit_score"] = float(cs_match.group(1))

    # Loan Amount: e.g. "loan amount $10,000", "loan 5000", "borrowing 20000"
    loan_match = re.search(r"(?:loan amount|loan|borrowing)(?:\s+of)?(?:\s*:)?\s*\$?([\d,]+)", q_lower)
    if loan_match:
        features["loan_amount"] = float(loan_match.group(1).replace(",", ""))

    # Years Employed: e.g. "5 years employed", "employed for 20 years", "2 years experience"
    emp_match = re.search(r"(\d+)\s*(?:years?|yrs?)\s*(?:employed|experience)|(?:employed|worked)\s*(?:for)?\s*(\d+)\s*(?:years?|yrs?)", q_lower)
    if emp_match:
        features["years_employed"] = float(emp_match.group(1) or emp_match.group(2))

    return {
        "locations": sorted(list(set(locations))),
        "metrics": sorted(list(set(metrics))),
        "dates": sorted(list(set(dates))),
        "features": features,
    }


def analyze(query: str) -> Dict[str, Any]:
    """Analyze query and return intent, confidence, entities, and route."""
    if not query or not str(query).strip():
        return {
            "processed_query": "",
            "intent": "UNKNOWN_INTENT",
            "confidence": 0.0,
            "entities": {"locations": [], "metrics": [], "dates": [], "features": {}},
            "route": "unknown",
            "message": "Empty query provided.",
        }

    q = str(query).strip()
    entities = extract_entities(q)
    model = _get_intent_model()

    intent = "UNKNOWN_INTENT"
    confidence = 0.5

    if model is not None:
        try:
            preds = model.predict([q])
            intent = str(preds[0])
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([q])[0]
                confidence = float(max(probs))
                # Low confidence threshold fallback
                if confidence < 0.40 and intent != "UNKNOWN_INTENT":
                    intent = "UNKNOWN_INTENT"
        except Exception as exc:
            logger.exception("Intent model error: %s", exc)
            intent = "UNKNOWN_INTENT"
    else:
        # Fallback heuristic
        q_low = q.lower()
        if any(w in q_low for w in ["boundary", "map", "gis", "district", "spatial", "telangana", "adilabad", "coordinates", "tier"]):
            intent = "GIS_VIEW"
            confidence = 0.85
        elif any(w in q_low for w in ["predict", "forecast", "estimate", "churn", "credit score", "loan amount"]):
            intent = "PREDICTION"
            confidence = 0.85
        elif any(w in q_low for w in ["metric", "rainfall", "revenue", "gdp", "inflation", "statistics", "count", "kpi"]):
            intent = "DATA_METRIC"
            confidence = 0.80
        elif any(w in q_low for w in ["what is", "explain", "who authored", "describe", "how does", "summarize"]):
            intent = "FAQ"
            confidence = 0.80
        else:
            intent = "UNKNOWN_INTENT"
            confidence = 0.30

    # Route mapping for downstream compatibility
    route_map = {
        "GIS_VIEW": "gis",
        "PREDICTION": "ml",
        "DATA_METRIC": "rag",
        "FAQ": "rag",
        "UNKNOWN_INTENT": "unknown",
    }
    route = route_map.get(intent, "unknown")

    return {
        "processed_query": q,
        "intent": intent,
        "confidence": confidence,
        "entities": entities,
        "route": route,
    }


def generate_structured_query(query: str) -> Dict[str, Any]:
    """Convert natural language query to deterministic backend / API request JSON payload."""
    t0 = time.perf_counter()
    analysis = analyze(query)
    intent = analysis["intent"]
    entities = analysis["entities"]
    locs = entities["locations"]
    metrics = entities["metrics"]
    dates = entities["dates"]
    features = entities["features"]

    payload: Dict[str, Any] = {
        "status": "success",
        "intent": intent,
        "raw_query": query,
        "target_service": "",
        "request_schema": {},
        "execution_time_ms": 0.0,
    }

    if intent == "GIS_VIEW":
        payload["target_service"] = "GIS_SPATIAL_SERVICE"
        primary_loc = locs[0] if locs else ""
        payload["request_schema"] = {
            "endpoint": "/api/v1/spatial/query" if primary_loc else "/api/v1/spatial/boundaries/1",
            "method": "GET",
            "params": {"q": primary_loc, "state": "Telangana"} if primary_loc else {"tier": 1},
        }
    elif intent == "PREDICTION":
        payload["target_service"] = "ML_INFERENCE_SERVICE"
        # Standardized schema matching PredictionPayload
        base_features = {
            "age": features.get("age", 35.0),
            "income": features.get("income", 50000.0),
            "credit_score": features.get("credit_score", 680.0),
            "loan_amount": features.get("loan_amount", 10000.0),
            "years_employed": features.get("years_employed", 5.0),
            "education": "bachelor",
            "employment_status": "employed",
            "marital_status": "single",
            "housing_type": "rent",
            "application_date": dates[0] if dates else "2026-09-21",
        }
        payload["request_schema"] = {
            "endpoint": "/predict",
            "method": "POST",
            "payload": base_features,
        }
    elif intent == "DATA_METRIC":
        payload["target_service"] = "BACKEND_DATA_SERVICE"
        payload["request_schema"] = {
            "endpoint": "/process",
            "method": "POST",
            "payload": {
                "profile": "company_financials",
                "filters": {
                    "locations": locs,
                    "metrics": metrics,
                    "dates": dates,
                },
            },
        }
    elif intent == "FAQ":
        payload["target_service"] = "RAG_KNOWLEDGE_SERVICE"
        payload["request_schema"] = {
            "endpoint": "/api/v1/ai/query",
            "method": "POST",
            "payload": {"query": query, "k": 3},
        }
    else:  # UNKNOWN_INTENT
        payload["target_service"] = "FALLBACK_HANDLER"
        payload["request_schema"] = {
            "status": "unrecognized",
            "message": "Query could not be mapped to an actionable domain. Please formulate an economic, spatial, or prediction query.",
        }

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    payload["execution_time_ms"] = round(elapsed_ms, 2)
    return payload


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) using Levenshtein distance on words."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    r_len = len(ref_words)
    h_len = len(hyp_words)

    # DP table
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1].lower() == hyp_words[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1] + 1
                ins = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                dp[i][j] = min(sub, ins, deletion)

    wer = dp[r_len][h_len] / float(r_len)
    return float(wer)


if __name__ == "__main__":
    test_queries = [
        "Show boundary map for Adilabad district in Telangana",
        "Predict default probability for a 35-year-old borrower with income $50,000 and credit score 680",
        "Show rainfall in Jagtial and Peddapalli for 2025",
        "What is CARIVIX AI platform?",
        "asdfghjkl random gibberish foo bar",
    ]
    for q in test_queries:
        res = analyze(q)
        print(f"\nQuery: {q}")
        print(f"Intent: {res['intent']} (Confidence: {res['confidence']:.2f})")
        print(f"Entities: {res['entities']}")
        sq = generate_structured_query(q)
        print(f"Generated JSON Schema in {sq['execution_time_ms']} ms -> Target: {sq['target_service']}")

