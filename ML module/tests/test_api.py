import types

import pytest
from fastapi.testclient import TestClient

from api import create_app
from src.model_service import ModelService


PAYLOAD = {
    "age": 43,
    "income": 67976.67,
    "credit_score": 694,
    "loan_amount": 14857.28,
    "years_employed": 19,
    "education": "Master",
    "employment_status": "Employed",
    "marital_status": "Married",
    "housing_type": "Rent",
    "application_date": "2024-09-21",
}


@pytest.fixture
def app_client():
    app = create_app()
    return TestClient(app)


def test_health_endpoint(app_client):
    response = app_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "CARIVIX AI Model Service"
    assert body["models_loaded"] >= 1


def test_model_listing_endpoint(app_client):
    response = app_client.get("/api/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert "models" in body
    assert len(body["models"]) >= 1
    assert body["models"][0]["status"] == "loaded"


def test_model_info_endpoint(app_client):
    response = app_client.get("/api/v1/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "classification"
    assert body["status"] == "loaded"
    assert body["supported_prediction_mode"] == "classification"


def test_valid_single_prediction(app_client):
    response = app_client.post("/api/v1/predict", json=PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "prediction" in body
    assert "model" in body


def test_invalid_prediction_input(app_client):
    bad_payload = dict(PAYLOAD)
    bad_payload["credit_score"] = 100
    response = app_client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"] == "Invalid request data."


def test_missing_required_fields(app_client):
    bad_payload = dict(PAYLOAD)
    bad_payload.pop("loan_amount")
    response = app_client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"] == "Invalid request data."


def test_batch_prediction(app_client):
    response = app_client.post("/api/v1/predict/batch", json={"records": [PAYLOAD, PAYLOAD]})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_no_model_scenario(tmp_path):
    empty_service = ModelService(models_dir=str(tmp_path))
    app = create_app(empty_service)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"

    predict_response = client.post("/api/v1/predict", json=PAYLOAD)
    assert predict_response.status_code == 503
    assert predict_response.json()["detail"] == "No trained model is currently available."


def test_model_loading_failure(tmp_path):
    bad_dir = tmp_path / "broken_models"
    bad_dir.mkdir()
    (bad_dir / "broken.pkl").write_bytes(b"not-a-valid-model")

    service = ModelService(models_dir=str(bad_dir))
    app = create_app(service)
    client = TestClient(app)

    response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert response.json()["models"] == []


def test_prediction_failure(monkeypatch):
    app = create_app()
    client = TestClient(app)

    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(app.state.model_service, "predict", raise_runtime_error)
    response = client.post("/api/v1/predict", json=PAYLOAD)
    assert response.status_code == 500
    assert response.json()["detail"] == "Model inference failed"
