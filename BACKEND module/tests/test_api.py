import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_sources_returns_all_categories():
    response = client.get("/sources")
    assert response.status_code == 200
    assert "csv_sources" in response.json()


def test_process_valid_request_returns_200():
    response = client.post("/process", json={
        "records": [
            {"Company Name": "Acme", "Revenue": 100.0, "Region": "South", "Report Date": "2026-01-05"},
            {"Company Name": "Beta", "Revenue": 250.0, "Region": "north", "Report Date": "2026-02-10"},
        ],
        "profile": "company_financials",
    })
    assert response.status_code == 200
    assert response.json()["row_count"] == 2


def test_process_unknown_profile_returns_400():
    response = client.post("/process", json={"records": [{"a": 1}], "profile": "nonexistent_profile"})
    assert response.status_code == 400


def test_process_empty_records_returns_400():
    response = client.post("/process", json={"records": [], "profile": "company_financials"})
    assert response.status_code == 400


def test_process_missing_field_returns_422():
    response = client.post("/process", json={"records": [{"a": 1}]})
    assert response.status_code == 422