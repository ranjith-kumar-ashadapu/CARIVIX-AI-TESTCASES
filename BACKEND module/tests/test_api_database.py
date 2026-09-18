import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database"))
import pytest
from fastapi.testclient import TestClient
from api import app
from database_service import DatabaseService

client = TestClient(app)

TEST_TABLES = ["test_items", "filter_test", "page_test"]


@pytest.fixture(autouse=True)
def clean_test_tables():
    with DatabaseService("carivix_api.db") as db:
        for table in TEST_TABLES:
            db.db.drop_table(table, if_exists=True)
    yield


def test_store_and_retrieve_round_trip():
    response = client.post("/store", json={
        "table": "test_items",
        "records": [{"name": "widget", "price": 9.99}],
    })
    assert response.status_code == 200
    assert response.json()["rows_written"] == 1

    response = client.get("/retrieve/test_items")
    assert response.status_code == 200
    assert response.json()["row_count"] >= 1


def test_store_empty_records_returns_400():
    response = client.post("/store", json={"table": "test_items", "records": []})
    assert response.status_code == 400


def test_retrieve_with_filter():
    client.post("/store", json={
        "table": "filter_test",
        "records": [{"category": "a", "value": 1}, {"category": "b", "value": 2}],
    })
    response = client.get("/retrieve/filter_test?filter_field=category&filter_value=a")
    assert response.status_code == 200
    assert response.json()["row_count"] == 1


def test_retrieve_with_pagination():
    client.post("/store", json={
        "table": "page_test",
        "records": [{"n": 1}, {"n": 2}, {"n": 3}],
    })
    response = client.get("/retrieve/page_test?limit=1&offset=1")
    assert response.status_code == 200
    assert response.json()["row_count"] == 1