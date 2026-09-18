"""
ML Model Inference API Tests
==============================

Maps to test cases:
  TC-ML-01  Feature Engineering Pipeline      (input validation / schema acceptance)
  TC-ML-02  Model Inference Endpoint          (HTTP 200 + score + latency < 300 ms)
  TC-ML-06  Model Boundary & Negative Inputs  (HTTP 422 for out-of-range values)
  TC-ML-07  Baseline Experiment Reproducibility (smoke – model metadata + determinism)

Extended scenarios: TC-EXT-ML-01 .. TC-EXT-ML-06

Service under test : http://127.0.0.1:8001  (ML module/api.py  – loan approval classifier)
Playwright fixture : ``ml_api`` (APIRequestContext) from conftest.py

Available trained models (ML module/models/):
  DecisionTree, GradientBoosting, KNN, LogisticRegression,
  NaiveBayes, RandomForest, SVM, XGBoost

Run:
    pytest carivix_tests/ml/test_ml_model_api.py -v
    pytest carivix_tests/ml/test_ml_model_api.py -m smoke -v
"""

from __future__ import annotations

import time

import pytest
from playwright.sync_api import APIRequestContext


# ---------------------------------------------------------------------------
# Representative prediction payload (valid – should return HTTP 200)
# ---------------------------------------------------------------------------
VALID_PAYLOAD: dict = {
    "age": 43,
    "income": 67_976.67,
    "credit_score": 694,
    "loan_amount": 14_857.28,
    "years_employed": 19,
    "education": "Master",
    "employment_status": "Employed",
    "marital_status": "Married",
    "housing_type": "Rent",
    "application_date": "2024-09-21",
}

# Boundary values – edge of the valid domain
BOUNDARY_PAYLOAD: dict = {
    "age": 18,           # minimum realistic
    "income": 0.0,       # ge=0 → valid at 0
    "credit_score": 300, # ge=300 → minimum allowed
    "loan_amount": 0.0,
    "years_employed": 0,
    "education": "High School",
    "employment_status": "Unemployed",
    "marital_status": "Single",
    "housing_type": "Own",
    "application_date": "2020-01-01",
}


# ===========================================================================
# TC-ML-07 (smoke) – Service health & model availability
# ===========================================================================

@pytest.mark.ml
@pytest.mark.smoke
def test_ml07_health_endpoint_returns_200(ml_api: APIRequestContext):
    """TC-ML-07 – GET /health responds HTTP 200."""
    response = ml_api.get("/health")
    assert response.status == 200, f"Expected 200, got {response.status}"


@pytest.mark.ml
@pytest.mark.smoke
def test_ml07_health_body_contains_required_fields(ml_api: APIRequestContext):
    """TC-ML-07 – /health body includes 'status', 'service', 'models_loaded'."""
    body = ml_api.get("/health").json()
    for field in ("status", "service", "models_loaded"):
        assert field in body, f"Missing field '{field}' in health response: {body}"


@pytest.mark.ml
@pytest.mark.smoke
def test_ml07_health_service_name_correct(ml_api: APIRequestContext):
    """TC-ML-07 – Service identifies itself as 'CARIVIX AI Model Service'."""
    body = ml_api.get("/health").json()
    assert body["service"] == "CARIVIX AI Model Service", (
        f"Unexpected service name: {body['service']}"
    )


@pytest.mark.ml
@pytest.mark.smoke
def test_ml07_models_loaded_at_least_one(ml_api: APIRequestContext):
    """TC-ML-07 – At least one trained model is loaded (8 pkl files are present)."""
    body = ml_api.get("/health").json()
    assert body["models_loaded"] >= 1, (
        f"No models loaded. Health response: {body}"
    )


@pytest.mark.ml
@pytest.mark.smoke
def test_ml07_health_status_is_healthy(ml_api: APIRequestContext):
    """TC-ML-07 – Health status is 'healthy' (not 'degraded') when models are present."""
    body = ml_api.get("/health").json()
    if body["models_loaded"] == 0:
        pytest.skip("No models loaded – cannot verify 'healthy' status")
    assert body["status"] == "healthy", f"Service degraded: {body}"


# ===========================================================================
# TC-ML-01  – Models list & metadata (Feature Engineering / model readiness)
# ===========================================================================

@pytest.mark.ml
def test_ml01_models_list_endpoint_returns_200(ml_api: APIRequestContext):
    """TC-ML-01 – GET /api/v1/models returns 200."""
    response = ml_api.get("/api/v1/models")
    assert response.status == 200, f"Expected 200, got {response.status}"


@pytest.mark.ml
def test_ml01_models_list_contains_models_key(ml_api: APIRequestContext):
    """TC-ML-01 – /api/v1/models body has a 'models' list."""
    body = ml_api.get("/api/v1/models").json()
    assert "models" in body, f"Missing 'models' key: {body}"
    assert isinstance(body["models"], list)


@pytest.mark.ml
def test_ml01_models_each_have_required_fields(ml_api: APIRequestContext, ml_has_models: bool):
    """TC-ML-01 – Each model entry contains 'name', 'type', 'status'."""
    if not ml_has_models:
        pytest.skip("No models loaded")
    models = ml_api.get("/api/v1/models").json()["models"]
    for m in models:
        for key in ("name", "type", "status"):
            assert key in m, f"Model entry missing '{key}': {m}"


@pytest.mark.ml
def test_ml01_all_loaded_models_have_status_loaded(ml_api: APIRequestContext, ml_has_models: bool):
    """TC-ML-01 – Every model in the list reports status == 'loaded'."""
    if not ml_has_models:
        pytest.skip("No models loaded")
    models = ml_api.get("/api/v1/models").json()["models"]
    for m in models:
        assert m["status"] == "loaded", f"Model {m['name']} has status {m['status']}"


@pytest.mark.ml
def test_ml01_model_info_endpoint_returns_200(ml_api: APIRequestContext, ml_has_models: bool):
    """TC-ML-01 – GET /api/v1/model/info → 200 when models are loaded."""
    if not ml_has_models:
        pytest.skip("No models loaded")
    response = ml_api.get("/api/v1/model/info")
    assert response.status == 200, f"Expected 200, got {response.status}"


@pytest.mark.ml
def test_ml01_model_info_fields_populated(ml_api: APIRequestContext, ml_has_models: bool):
    """TC-ML-01 – Model info contains name, type, task_type, algorithm, status."""
    if not ml_has_models:
        pytest.skip("No models loaded")
    body = ml_api.get("/api/v1/model/info").json()
    for field in ("name", "type", "task_type", "algorithm", "status"):
        assert field in body, f"Missing field '{field}' in model info: {body}"
        assert body[field], f"Field '{field}' is empty in model info"


@pytest.mark.ml
def test_ml01_model_info_no_model_returns_503(ml_api: APIRequestContext, ml_has_models: bool):
    """TC-ML-01 – /api/v1/model/info → 503 when no models loaded (degraded state)."""
    if ml_has_models:
        pytest.skip("Models are loaded – cannot test 503 path in this environment")
    response = ml_api.get("/api/v1/model/info", fail_on_status_code=False)
    assert response.status == 503


# ===========================================================================
# TC-ML-02  – Single prediction endpoint
# ===========================================================================

@pytest.mark.ml
def test_ml02_predict_valid_payload_returns_200(
    ml_api: APIRequestContext, skip_if_no_models
):
    """TC-ML-02 – POST /api/v1/predict with valid payload → HTTP 200."""
    response = ml_api.post("/api/v1/predict", data=VALID_PAYLOAD)
    assert response.status == 200, f"Expected 200, got {response.status}: {response.text()}"


@pytest.mark.ml
def test_ml02_predict_response_body_structure(
    ml_api: APIRequestContext, skip_if_no_models
):
    """TC-ML-02 – Prediction response contains 'success', 'model', 'prediction'."""
    body = ml_api.post("/api/v1/predict", data=VALID_PAYLOAD).json()
    assert body.get("success") is True
    assert "prediction" in body
    assert "model" in body


@pytest.mark.ml
def test_ml02_predict_response_latency_under_300ms(
    ml_api: APIRequestContext, skip_if_no_models
):
    """TC-ML-02 – Prediction completes in < 300 ms (SLA from TC-ML-02)."""
    start = time.perf_counter()
    response = ml_api.post("/api/v1/predict", data=VALID_PAYLOAD)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert response.status == 200
    assert elapsed_ms < 300, f"Inference too slow: {elapsed_ms:.1f} ms (limit 300 ms)"


@pytest.mark.ml
def test_ml02_predict_boundary_values_accepted(
    ml_api: APIRequestContext, skip_if_no_models
):
    """TC-ML-02 – Boundary-valid payload (min credit_score=300, income=0) accepted."""
    response = ml_api.post("/api/v1/predict", data=BOUNDARY_PAYLOAD)
    assert response.status == 200, f"Boundary payload rejected: {response.text()}"


# ===========================================================================
# TC-ML-06  – Model boundary & negative inputs (schema validation)
# ===========================================================================

@pytest.mark.ml
def test_ml06_credit_score_below_300_returns_422(ml_api: APIRequestContext):
    """TC-ML-06 – credit_score < 300 (min boundary violation) → 422."""
    bad = dict(VALID_PAYLOAD, credit_score=100)
    response = ml_api.post("/api/v1/predict", data=bad, fail_on_status_code=False)
    assert response.status == 422, f"Expected 422, got {response.status}"


@pytest.mark.ml
def test_ml06_credit_score_above_850_returns_422(ml_api: APIRequestContext):
    """TC-ML-06 – credit_score > 850 (max boundary violation) → 422."""
    bad = dict(VALID_PAYLOAD, credit_score=900)
    response = ml_api.post("/api/v1/predict", data=bad, fail_on_status_code=False)
    assert response.status == 422


@pytest.mark.ml
def test_ml06_negative_income_returns_422(ml_api: APIRequestContext):
    """TC-ML-06 – income < 0 → 422."""
    bad = dict(VALID_PAYLOAD, income=-1.0)
    response = ml_api.post("/api/v1/predict", data=bad, fail_on_status_code=False)
    assert response.status == 422


@pytest.mark.ml
def test_ml06_zero_age_returns_422(ml_api: APIRequestContext):
    """TC-ML-06 – age = 0 (gt=0 violated) → 422."""
    bad = dict(VALID_PAYLOAD, age=0)
    response = ml_api.post("/api/v1/predict", data=bad, fail_on_status_code=False)
    assert response.status == 422


@pytest.mark.ml
def test_ml06_negative_loan_amount_returns_422(ml_api: APIRequestContext):
    """TC-ML-06 – loan_amount < 0 (ge=0 violated) → 422."""
    bad = dict(VALID_PAYLOAD, loan_amount=-500.0)
    response = ml_api.post("/api/v1/predict", data=bad, fail_on_status_code=False)
    assert response.status == 422


@pytest.mark.ml
def test_ml06_missing_required_field_loan_amount_returns_422(ml_api: APIRequestContext):
    """TC-ML-06 – Omitting required field 'loan_amount' → 422."""
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "loan_amount"}
    response = ml_api.post("/api/v1/predict", data=bad, fail_on_status_code=False)
    assert response.status == 422


@pytest.mark.ml
def test_ml06_missing_required_field_education_returns_422(ml_api: APIRequestContext):
    """TC-ML-06 – Omitting 'education' field → 422."""
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "education"}
    response = ml_api.post("/api/v1/predict", data=bad, fail_on_status_code=False)
    assert response.status == 422


@pytest.mark.ml
def test_ml06_422_error_body_structure(ml_api: APIRequestContext):
    """TC-ML-06 – 422 response includes 'success': false and 'error' field."""
    bad = dict(VALID_PAYLOAD, credit_score=100)
    body = ml_api.post("/api/v1/predict", data=bad, fail_on_status_code=False).json()
    assert body.get("success") is False
    assert "error" in body


@pytest.mark.ml
def test_ml06_empty_string_education_returns_422(ml_api: APIRequestContext):
    """TC-ML-06 – Empty string for 'education' (min_length=1 violated) → 422."""
    bad = dict(VALID_PAYLOAD, education="")
    response = ml_api.post("/api/v1/predict", data=bad, fail_on_status_code=False)
    assert response.status == 422


# ===========================================================================
# TC-EXT-ML – Extended ML inference scenarios
# ===========================================================================

@pytest.mark.ml
def test_ext_ml01_batch_prediction_correct_count(
    ml_api: APIRequestContext, skip_if_no_models
):
    """TC-EXT-ML-01 – POST /api/v1/predict/batch with N records → count == N."""
    batch = {"records": [VALID_PAYLOAD, VALID_PAYLOAD, VALID_PAYLOAD]}
    response = ml_api.post("/api/v1/predict/batch", data=batch)
    assert response.status == 200, f"Batch failed: {response.text()}"
    body = response.json()
    assert body["success"] is True
    assert body["count"] == 3
    assert len(body["predictions"]) == 3


@pytest.mark.ml
def test_ext_ml02_batch_empty_records_returns_422(ml_api: APIRequestContext):
    """TC-EXT-ML-02 – /api/v1/predict/batch with empty list → 422."""
    response = ml_api.post(
        "/api/v1/predict/batch", data={"records": []}, fail_on_status_code=False
    )
    assert response.status == 422


@pytest.mark.ml
def test_ext_ml03_prediction_determinism_same_input(
    ml_api: APIRequestContext, skip_if_no_models
):
    """TC-EXT-ML-03 – Same payload always returns the same prediction (deterministic model)."""
    r1 = ml_api.post("/api/v1/predict", data=VALID_PAYLOAD).json()
    r2 = ml_api.post("/api/v1/predict", data=VALID_PAYLOAD).json()
    assert r1["prediction"] == r2["prediction"], (
        f"Non-deterministic result: {r1['prediction']} vs {r2['prediction']}"
    )
    assert r1["model"] == r2["model"]


@pytest.mark.ml
def test_ext_ml04_predict_no_model_returns_503(ml_api: APIRequestContext, ml_has_models: bool):
    """TC-EXT-ML-04 – /api/v1/predict returns 503 when no models are loaded (degraded state)."""
    if ml_has_models:
        pytest.skip("Models loaded – 503 path not reachable in this environment")
    response = ml_api.post("/api/v1/predict", data=VALID_PAYLOAD, fail_on_status_code=False)
    assert response.status == 503


@pytest.mark.ml
def test_ext_ml05_batch_response_model_name_consistent(
    ml_api: APIRequestContext, skip_if_no_models
):
    """TC-EXT-ML-05 – Batch response 'model' field is a non-empty string."""
    batch = {"records": [VALID_PAYLOAD, VALID_PAYLOAD]}
    body = ml_api.post("/api/v1/predict/batch", data=batch).json()
    assert isinstance(body.get("model"), str)
    assert len(body["model"]) > 0


@pytest.mark.ml
def test_ext_ml06_single_and_batch_predictions_agree(
    ml_api: APIRequestContext, skip_if_no_models
):
    """TC-EXT-ML-06 – Single and batch predictions for the same record return the same result."""
    single_body = ml_api.post("/api/v1/predict", data=VALID_PAYLOAD).json()
    batch_body  = ml_api.post("/api/v1/predict/batch", data={"records": [VALID_PAYLOAD]}).json()
    assert single_body["prediction"] == batch_body["predictions"][0], (
        f"Single={single_body['prediction']} vs Batch={batch_body['predictions'][0]}"
    )


# ===========================================================================
# TC-ML-07 (smoke) – Baseline Reproducibility via model metadata
# ===========================================================================

@pytest.mark.ml
@pytest.mark.smoke
def test_ml07_model_info_algorithm_is_populated(ml_api: APIRequestContext, ml_has_models: bool):
    """TC-ML-07 (smoke) – 'algorithm' field in model info is a non-empty string."""
    if not ml_has_models:
        pytest.skip("No models loaded")
    body = ml_api.get("/api/v1/model/info").json()
    assert isinstance(body.get("algorithm"), str) and body["algorithm"], (
        f"algorithm field is empty: {body}"
    )


@pytest.mark.ml
@pytest.mark.smoke
def test_ml07_model_type_is_classification(ml_api: APIRequestContext, ml_has_models: bool):
    """TC-ML-07 (smoke) – Model type confirms it is a classification model."""
    if not ml_has_models:
        pytest.skip("No models loaded")
    body = ml_api.get("/api/v1/model/info").json()
    assert body.get("type") == "classification", (
        f"Expected type='classification', got: {body.get('type')}"
    )


@pytest.mark.ml
@pytest.mark.smoke
def test_ml07_model_supported_mode_is_classification(
    ml_api: APIRequestContext, ml_has_models: bool
):
    """TC-ML-07 (smoke) – supported_prediction_mode matches 'classification'."""
    if not ml_has_models:
        pytest.skip("No models loaded")
    body = ml_api.get("/api/v1/model/info").json()
    assert body.get("supported_prediction_mode") == "classification"


@pytest.mark.ml
@pytest.mark.smoke
def test_ml07_predict_returns_binary_classification_output(
    ml_api: APIRequestContext, skip_if_no_models
):
    """TC-ML-07 (smoke) – Prediction output is 0 or 1 (binary classification)."""
    body = ml_api.post("/api/v1/predict", data=VALID_PAYLOAD).json()
    pred = body.get("prediction")
    assert pred in (0, 1), f"Unexpected prediction value: {pred} (expected 0 or 1)"
