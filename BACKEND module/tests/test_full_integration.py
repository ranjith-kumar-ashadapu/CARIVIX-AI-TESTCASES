import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database"))
import pytest
from fastapi.testclient import TestClient
from api import app
from database_service import DatabaseService

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_table():
    with DatabaseService("carivix_api.db") as db:
        db.db.drop_table("full_integration_test", if_exists=True)
    yield


def test_validate_then_pipeline_then_retrieve():
    records = [
        {"Company Name": "Acme", "Revenue": 100.0, "Region": "South", "Report Date": "2026-01-05"},
        {"Company Name": "Beta", "Revenue": 250.0, "Region": "north", "Report Date": "2026-02-10"},
    ]

    validate_response = client.post("/validate", json={
        "records": records,
        "rules": [{"name": "Revenue", "dtype": "float", "required": True, "min_value": 0}],
    })
    assert validate_response.status_code == 200

    pipeline_response = client.post("/pipeline", json={
        "records": records,
        "profile": "company_financials",
        "table": "full_integration_test",
    })
    assert pipeline_response.status_code == 200
    assert pipeline_response.json()["rows_written"] == 2

    retrieve_response = client.get("/retrieve/full_integration_test")
    assert retrieve_response.status_code == 200
    assert retrieve_response.json()["row_count"] == 2