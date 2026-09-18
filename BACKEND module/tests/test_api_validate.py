import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_validate_flags_missing_and_out_of_range_values():
    response = client.post("/validate", json={
        "records": [
            {"district": "Hyderabad", "unemployment_rate": 5.2},
            {"district": None, "unemployment_rate": 6.0},
            {"district": "Hyderabad", "unemployment_rate": 150.0},
        ],
        "rules": [
            {"name": "district", "dtype": "str", "required": True},
            {"name": "unemployment_rate", "dtype": "float", "required": True, "min_value": 0, "max_value": 100},
        ],
    })
    assert response.status_code == 200
    body = response.json()
    assert body["total_rows"] == 3
    assert body["valid_rows"] == 1
    assert body["is_valid"] is False


def test_validate_empty_records_returns_400():
    response = client.post("/validate", json={"records": [], "rules": []})
    assert response.status_code == 400