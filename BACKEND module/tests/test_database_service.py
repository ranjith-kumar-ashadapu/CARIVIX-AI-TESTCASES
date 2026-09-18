import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "database"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_processing"))
import pytest
from database_service import DatabaseService, DatabaseServiceError
import pandas as pd


def make_service():
    svc = DatabaseService(":memory:")
    svc.db.create_table("items", {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "name": "TEXT NOT NULL",
    })
    return svc


def test_write_single_record():
    svc = make_service()
    count = svc.write("items", [{"name": "apple"}])
    assert count == 1


def test_write_multiple_records():
    svc = make_service()
    count = svc.write("items", [{"name": "apple"}, {"name": "banana"}])
    assert count == 2


def test_read_returns_written_records():
    svc = make_service()
    svc.write("items", [{"name": "apple"}])
    rows = svc.read("items")
    assert len(rows) == 1
    assert rows[0]["name"] == "apple"


def test_read_with_filter():
    svc = make_service()
    svc.write("items", [{"name": "apple"}, {"name": "banana"}])
    rows = svc.read("items", where={"name": "banana"})
    assert len(rows) == 1


def test_write_to_missing_table_raises_service_error():
    svc = make_service()
    with pytest.raises(DatabaseServiceError):
        svc.write("nonexistent_table", [{"name": "test"}])


def test_context_manager_closes_connection():
    with make_service() as svc:
        svc.write("items", [{"name": "apple"}])
    assert True


def test_write_handles_timestamp_values():
    svc = make_service()
    svc.db.create_table("dated_items", {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "name": "TEXT",
        "date": "TEXT",
    })
    svc.write("dated_items", [{"name": "apple", "date": pd.Timestamp("2026-01-05")}])
    rows = svc.read("dated_items")
    assert rows[0]["date"] == "2026-01-05 00:00:00"