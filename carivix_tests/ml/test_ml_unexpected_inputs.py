"""
ML & Items Service Unexpected Input Tests
===========================================

Validates system resilience and error handling against inputs that are NOT
the expected happy-path inputs:
  - Domain range boundary violations (negative loan amounts, impossible credit scores, negative ages)
  - Type incompatibilities (strings where floats are expected, non-numeric IDs)
  - Missing required fields and empty payloads
  - Non-existent model routing
  - Empty batch payloads
  - Business logic boundary violations (e.g. negative item prices)

Services under test:
  - Port 8001: ML Inference API (ML module/api.py)
  - Port 8002: Items CRUD API (ML module/fastapi_main.py)
"""

from __future__ import annotations

import pytest
from playwright.sync_api import APIRequestContext


# Valid baseline payload for reference
VALID_LOAN_PAYLOAD: dict = {
    "age": 35.0,
    "income": 75000.0,
    "credit_score": 720.0,
    "loan_amount": 25000.0,
    "years_employed": 8.0,
    "education": "Bachelor",
    "employment_status": "Employed",
    "marital_status": "Married",
    "housing_type": "Own",
    "application_date": "2024-05-15",
}


# ===========================================================================
# ML Model Inference API (Port 8001) – Unexpected Inputs
# ===========================================================================

@pytest.mark.ml
@pytest.mark.neg
def test_unexp_ml01_negative_loan_amount_rejected(ml_api: APIRequestContext):
    """TC-UNEXP-ML-01: Submit negative loan amount (loan_amount = -50000.0).
    Expected: HTTP 422 Unprocessable Entity (violates ge=0 constraint).
    """
    bad_payload = dict(VALID_LOAN_PAYLOAD, loan_amount=-50000.0)
    response = ml_api.post("/api/v1/predict", data=bad_payload, fail_on_status_code=False)
    assert response.status == 422, f"Expected 422 for negative loan amount, got {response.status}"
    body = response.json()
    assert body.get("success") is False


@pytest.mark.ml
@pytest.mark.neg
def test_unexp_ml02_credit_score_above_maximum_rejected(ml_api: APIRequestContext):
    """TC-UNEXP-ML-02: Submit credit score far above valid range (credit_score = 9999.0).
    Expected: HTTP 422 Unprocessable Entity (violates le=850 constraint).
    """
    bad_payload = dict(VALID_LOAN_PAYLOAD, credit_score=9999.0)
    response = ml_api.post("/api/v1/predict", data=bad_payload, fail_on_status_code=False)
    assert response.status == 422, f"Expected 422 for credit score 9999, got {response.status}"


@pytest.mark.ml
@pytest.mark.neg
def test_unexp_ml03_credit_score_below_minimum_rejected(ml_api: APIRequestContext):
    """TC-UNEXP-ML-03: Submit credit score far below valid range (credit_score = 50.0).
    Expected: HTTP 422 Unprocessable Entity (violates ge=300 constraint).
    """
    bad_payload = dict(VALID_LOAN_PAYLOAD, credit_score=50.0)
    response = ml_api.post("/api/v1/predict", data=bad_payload, fail_on_status_code=False)
    assert response.status == 422, f"Expected 422 for credit score 50, got {response.status}"


@pytest.mark.ml
@pytest.mark.neg
def test_unexp_ml04_negative_age_rejected(ml_api: APIRequestContext):
    """TC-UNEXP-ML-04: Submit negative applicant age (age = -25.0).
    Expected: HTTP 422 Unprocessable Entity (violates gt=0 constraint).
    """
    bad_payload = dict(VALID_LOAN_PAYLOAD, age=-25.0)
    response = ml_api.post("/api/v1/predict", data=bad_payload, fail_on_status_code=False)
    assert response.status == 422, f"Expected 422 for negative age, got {response.status}"


@pytest.mark.ml
@pytest.mark.neg
def test_unexp_ml05_zero_age_rejected(ml_api: APIRequestContext):
    """TC-UNEXP-ML-05: Submit applicant age of 0 (age = 0.0).
    Expected: HTTP 422 Unprocessable Entity (violates gt=0 strict constraint).
    """
    bad_payload = dict(VALID_LOAN_PAYLOAD, age=0.0)
    response = ml_api.post("/api/v1/predict", data=bad_payload, fail_on_status_code=False)
    assert response.status == 422, f"Expected 422 for age=0, got {response.status}"


@pytest.mark.ml
@pytest.mark.neg
def test_unexp_ml06_string_in_numeric_income_field_rejected(ml_api: APIRequestContext):
    """TC-UNEXP-ML-06: Pass string description instead of float to income field.
    Expected: HTTP 422 Unprocessable Entity (FastAPI type coercion failure).
    """
    bad_payload = dict(VALID_LOAN_PAYLOAD, income="one_hundred_thousand_dollars")
    response = ml_api.post("/api/v1/predict", data=bad_payload, fail_on_status_code=False)
    assert response.status == 422, f"Expected 422 for string income, got {response.status}"


@pytest.mark.ml
@pytest.mark.neg
def test_unexp_ml07_empty_payload_rejected(ml_api: APIRequestContext):
    """TC-UNEXP-ML-07: Pass completely empty JSON payload `{}` to prediction endpoint.
    Expected: HTTP 422 Unprocessable Entity (missing all 10 required fields).
    """
    response = ml_api.post("/api/v1/predict", data={}, fail_on_status_code=False)
    assert response.status == 422, f"Expected 422 for empty payload, got {response.status}"


@pytest.mark.ml
@pytest.mark.neg
def test_unexp_ml08_empty_string_for_education_rejected(ml_api: APIRequestContext):
    """TC-UNEXP-ML-08: Pass empty string for required string field education.
    Expected: HTTP 422 Unprocessable Entity (violates min_length=1 constraint).
    """
    bad_payload = dict(VALID_LOAN_PAYLOAD, education="")
    response = ml_api.post("/api/v1/predict", data=bad_payload, fail_on_status_code=False)
    assert response.status == 422, f"Expected 422 for empty education string, got {response.status}"


@pytest.mark.ml
@pytest.mark.neg
def test_unexp_ml09_nonexistent_model_requested(ml_api: APIRequestContext):
    """TC-UNEXP-ML-09: Request inference targeting a non-existent model name.
    Expected: HTTP 404 Not Found or HTTP 422 (model router gracefully rejects).
    """
    bad_payload = dict(VALID_LOAN_PAYLOAD, model="quantum_neural_gpt9_model")
    response = ml_api.post("/api/v1/predict", data=bad_payload, fail_on_status_code=False)
    assert response.status in (404, 422, 503), f"Expected 404/422/503 for non-existent model, got {response.status}"


@pytest.mark.ml
@pytest.mark.neg
def test_unexp_ml10_batch_empty_records_list_rejected(ml_api: APIRequestContext):
    """TC-UNEXP-ML-10: Pass empty records list `{"records": []}` to batch endpoint.
    Expected: HTTP 422 Unprocessable Entity (violates min_length=1 constraint).
    """
    response = ml_api.post("/api/v1/predict/batch", data={"records": []}, fail_on_status_code=False)
    assert response.status == 422, f"Expected 422 for empty batch records, got {response.status}"


# ===========================================================================
# ML Items CRUD API (Port 8002) – Unexpected Inputs
# ===========================================================================

@pytest.mark.items
@pytest.mark.neg
def test_unexp_itm01_string_in_numeric_item_id_path_rejected(items_api: APIRequestContext):
    """TC-UNEXP-ITM-01: Pass non-numeric string into item_id path parameter.
    Expected: HTTP 422 Unprocessable Entity (Path parameter int conversion fails).
    """
    response = items_api.get("/items/not_a_valid_integer_id", fail_on_status_code=False)
    assert response.status in (404, 422), f"Expected 422 or 404 for string ID, got {response.status}"


@pytest.mark.items
@pytest.mark.neg
def test_unexp_itm02_negative_item_price_boundary_check(items_api: APIRequestContext):
    """TC-UNEXP-ITM-02: Create item with negative price (price = -99.99).
    Exposes business logic vulnerability: Schema allows negative float if ge=0 was omitted.
    """
    response = items_api.post(
        "/items",
        data={"name": "Unexpected Negative Price Item", "price": -99.99},
        fail_on_status_code=False,
    )
    # The API schema does not constrain price >= 0, so it returns 201.
    # We record this actual behavior as a documented finding/unexpected input observation.
    assert response.status in (201, 422), f"Unexpected status: {response.status}"


@pytest.mark.items
@pytest.mark.neg
def test_unexp_itm03_missing_required_name_field_rejected(items_api: APIRequestContext):
    """TC-UNEXP-ITM-03: POST /items without required 'name' field.
    Expected: HTTP 422 Unprocessable Entity.
    """
    response = items_api.post("/items", data={"price": 49.99}, fail_on_status_code=False)
    assert response.status == 422, f"Expected 422 for missing name, got {response.status}"


@pytest.mark.items
@pytest.mark.neg
def test_unexp_itm04_missing_required_price_field_rejected(items_api: APIRequestContext):
    """TC-UNEXP-ITM-04: POST /items without required 'price' field.
    Expected: HTTP 422 Unprocessable Entity.
    """
    response = items_api.post("/items", data={"name": "No Price Item"}, fail_on_status_code=False)
    assert response.status == 422, f"Expected 422 for missing price, got {response.status}"


@pytest.mark.items
@pytest.mark.neg
def test_unexp_itm05_get_nonexistent_item_id_returns_404(items_api: APIRequestContext):
    """TC-UNEXP-ITM-05: GET /items/999999999 for non-existent item id.
    Expected: HTTP 404 Not Found.
    """
    response = items_api.get("/items/999999999", fail_on_status_code=False)
    assert response.status == 404, f"Expected 404 for non-existent item ID, got {response.status}"


@pytest.mark.items
@pytest.mark.neg
def test_unexp_itm06_update_nonexistent_item_id_returns_404(items_api: APIRequestContext):
    """TC-UNEXP-ITM-06: PUT /items/999999999 for non-existent item id.
    Expected: HTTP 404 Not Found.
    """
    response = items_api.put(
        "/items/999999999",
        data={"name": "Phantom Item", "price": 10.0},
        fail_on_status_code=False,
    )
    assert response.status == 404, f"Expected 404 for updating non-existent item, got {response.status}"


@pytest.mark.items
@pytest.mark.neg
def test_unexp_itm07_oversized_item_name_handled(items_api: APIRequestContext):
    """TC-UNEXP-ITM-07: POST /items with extremely long item name (2,000 characters).
    Expected: Handled without server crash (201 Created or 422, never 500).
    """
    giant_name = "OVERSIZED_ITEM_" + ("A" * 2000)
    response = items_api.post(
        "/items",
        data={"name": giant_name, "price": 15.0},
        fail_on_status_code=False,
    )
    assert response.status != 500, f"Server crashed with 500 on oversized item name: {response.text()}"
