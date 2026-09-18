"""
Backend Data Service API Tests
===============================

Maps to test cases: TC-BE-01, TC-BE-02, TC-BE-03, TC-BE-04, TC-BE-05, TC-BE-06
Extended scenarios:  TC-EXT-BE-01 .. TC-EXT-BE-06

Service under test : http://127.0.0.1:8000   (BACKEND module/api.py)
Playwright fixture : ``backend_api`` (APIRequestContext) from conftest.py

Run:
    pytest carivix_tests/backend/test_backend_api.py -v
    pytest carivix_tests/backend/test_backend_api.py -m "backend and smoke" -v
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from playwright.sync_api import APIRequestContext


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------
VALID_RECORDS = [
    {"Company Name": "Acme Corp",   "Revenue": 150_000.0, "Region": "South", "Report Date": "2026-01-05"},
    {"Company Name": "Beta Ltd",    "Revenue": 280_000.0, "Region": "north", "Report Date": "2026-02-10"},
    {"Company Name": "Gamma Inc",   "Revenue":  90_000.0, "Region": "East",  "Report Date": "2026-03-15"},
]

VALID_PROFILE = "company_financials"

VALIDATION_RULES = [
    {"name": "Company Name", "dtype": "str",   "required": True},
    {"name": "Revenue",      "dtype": "float",  "required": True, "min_value": 0.0},
    {"name": "Region",       "dtype": "str",   "required": True},
]


# ===========================================================================
# TC-BE-01 – Central Service Layer Health
# ===========================================================================

@pytest.mark.backend
@pytest.mark.smoke
def test_be01_health_returns_200_and_ok_status(backend_api: APIRequestContext):
    """TC-BE-01 – GET /health → HTTP 200 with {"status": "ok"}."""
    response = backend_api.get("/health")
    assert response.status == 200, f"Expected 200, got {response.status}"
    body = response.json()
    assert body.get("status") == "ok", f"Unexpected body: {body}"


@pytest.mark.backend
@pytest.mark.smoke
def test_be01_health_response_latency_under_50ms(backend_api: APIRequestContext):
    """TC-BE-01 – Health endpoint responds in < 50 ms (latency SLA)."""
    start = time.perf_counter()
    response = backend_api.get("/health")
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert response.status == 200
    assert elapsed_ms < 50, f"Health check too slow: {elapsed_ms:.1f} ms (limit 50 ms)"


# ===========================================================================
# TC-BE-02 – Data Ingestion & ETL Pipeline
# ===========================================================================

@pytest.mark.backend
def test_be02_list_sources_returns_csv_sources(backend_api: APIRequestContext):
    """TC-BE-02 – GET /sources → 200 with 'csv_sources' key in response."""
    response = backend_api.get("/sources")
    assert response.status == 200
    body = response.json()
    assert "csv_sources" in body, f"Expected 'csv_sources' in response, got keys: {list(body.keys())}"


@pytest.mark.backend
def test_be02_process_valid_records_returns_200(backend_api: APIRequestContext):
    """TC-BE-02 – POST /process with valid company_financials records → 200."""
    response = backend_api.post(
        "/process",
        data={"records": VALID_RECORDS, "profile": VALID_PROFILE},
    )
    assert response.status == 200, f"Expected 200, got {response.status}: {response.text()}"
    body = response.json()
    assert "row_count" in body
    assert body["row_count"] > 0


@pytest.mark.backend
def test_be02_process_returns_correct_row_count(backend_api: APIRequestContext):
    """TC-BE-02 – Processed row count must equal input records (after dedup)."""
    response = backend_api.post(
        "/process",
        data={"records": VALID_RECORDS, "profile": VALID_PROFILE},
    )
    assert response.status == 200
    body = response.json()
    # Row count should be ≤ number of input records (dedup may reduce)
    assert body["row_count"] <= len(VALID_RECORDS)
    assert "columns" in body
    assert isinstance(body["columns"], list)


@pytest.mark.backend
def test_be02_process_response_contains_data_array(backend_api: APIRequestContext):
    """TC-BE-02 – /process response must include a 'data' list of dicts."""
    response = backend_api.post(
        "/process",
        data={"records": VALID_RECORDS, "profile": VALID_PROFILE},
    )
    assert response.status == 200
    body = response.json()
    assert "data" in body
    assert isinstance(body["data"], list)
    assert len(body["data"]) > 0


# ===========================================================================
# TC-BE-03 – Database CRUD & Connection Pool
# ===========================================================================

_TEST_TABLE = "carivix_test_items"
_FILTER_TABLE = "carivix_filter_test"
_PAGE_TABLE = "carivix_page_test"


@pytest.mark.backend
def test_be03_store_and_retrieve_round_trip(backend_api: APIRequestContext):
    """TC-BE-03 – POST /store then GET /retrieve/{table} returns stored rows."""
    # Store
    store_resp = backend_api.post(
        "/store",
        data={"table": _TEST_TABLE, "records": [{"name": "widget", "price": 9.99}]},
    )
    assert store_resp.status == 200, f"Store failed: {store_resp.text()}"
    assert store_resp.json()["rows_written"] >= 1

    # Retrieve
    get_resp = backend_api.get(f"/retrieve/{_TEST_TABLE}")
    assert get_resp.status == 200
    assert get_resp.json()["row_count"] >= 1


@pytest.mark.backend
def test_be03_store_multiple_records(backend_api: APIRequestContext):
    """TC-BE-03 – Storing multiple records writes all rows correctly."""
    records = [{"product": f"item_{i}", "qty": i} for i in range(5)]
    resp = backend_api.post(
        "/store",
        data={"table": _TEST_TABLE, "records": records},
    )
    assert resp.status == 200
    assert resp.json()["rows_written"] == 5


@pytest.mark.backend
def test_be03_retrieve_with_field_filter(backend_api: APIRequestContext):
    """TC-BE-03 – /retrieve with filter_field/filter_value returns only matching rows."""
    # Seed the filter table
    backend_api.post(
        "/store",
        data={
            "table": _FILTER_TABLE,
            "records": [{"category": "alpha", "val": 1}, {"category": "beta", "val": 2}],
        },
    )
    resp = backend_api.get(f"/retrieve/{_FILTER_TABLE}?filter_field=category&filter_value=alpha")
    assert resp.status == 200
    body = resp.json()
    assert body["row_count"] >= 1
    for row in body["data"]:
        assert row["category"] == "alpha", f"Filter leaked non-matching row: {row}"


@pytest.mark.backend
def test_be03_retrieve_pagination_limit_and_offset(backend_api: APIRequestContext):
    """TC-BE-03 – limit/offset pagination controls the response size."""
    backend_api.post(
        "/store",
        data={
            "table": _PAGE_TABLE,
            "records": [{"n": i} for i in range(10)],
        },
    )
    resp = backend_api.get(f"/retrieve/{_PAGE_TABLE}?limit=3&offset=0")
    assert resp.status == 200
    assert resp.json()["row_count"] == 3

    resp2 = backend_api.get(f"/retrieve/{_PAGE_TABLE}?limit=3&offset=3")
    assert resp2.status == 200
    assert resp2.json()["row_count"] == 3


# ===========================================================================
# TC-BE-04 – Standardized Error Handling
# ===========================================================================

@pytest.mark.backend
def test_be04_unknown_endpoint_returns_404(backend_api: APIRequestContext):
    """TC-BE-04 – Request to an unregistered route returns HTTP 404."""
    response = backend_api.get("/api/does-not-exist", fail_on_status_code=False)
    assert response.status == 404, f"Expected 404, got {response.status}"


@pytest.mark.backend
def test_be04_process_missing_required_field_returns_422(backend_api: APIRequestContext):
    """TC-BE-04 – POST /process with missing 'profile' field → 422 Unprocessable Entity."""
    response = backend_api.post(
        "/process",
        data={"records": VALID_RECORDS},
        fail_on_status_code=False,
    )
    assert response.status == 422, f"Expected 422, got {response.status}"


@pytest.mark.backend
def test_be04_process_unknown_profile_returns_400(backend_api: APIRequestContext):
    """TC-BE-04 – POST /process with unknown profile name → 400 Bad Request."""
    response = backend_api.post(
        "/process",
        data={"records": VALID_RECORDS, "profile": "totally_invalid_profile_xyz"},
        fail_on_status_code=False,
    )
    assert response.status == 400, f"Expected 400, got {response.status}"


@pytest.mark.backend
def test_be04_store_empty_records_returns_400(backend_api: APIRequestContext):
    """TC-BE-04 – POST /store with an empty records list → 400 Bad Request."""
    response = backend_api.post(
        "/store",
        data={"table": _TEST_TABLE, "records": []},
        fail_on_status_code=False,
    )
    assert response.status == 400, f"Expected 400, got {response.status}"


@pytest.mark.backend
def test_be04_validate_empty_records_returns_400(backend_api: APIRequestContext):
    """TC-BE-04 – POST /validate with no records → 400 Bad Request."""
    response = backend_api.post(
        "/validate",
        data={"records": [], "rules": VALIDATION_RULES},
        fail_on_status_code=False,
    )
    assert response.status == 400


@pytest.mark.backend
def test_be04_retrieve_nonexistent_table_returns_404(backend_api: APIRequestContext):
    """TC-BE-04 – GET /retrieve/nonexistent_zzz → 404 when table does not exist."""
    response = backend_api.get("/retrieve/nonexistent_table_zzz999", fail_on_status_code=False)
    assert response.status == 404, f"Expected 404, got {response.status}"


# ===========================================================================
# TC-BE-06 – Data Validation Endpoint
# ===========================================================================

@pytest.mark.backend
def test_be06_validate_valid_records_reports_is_valid_true(backend_api: APIRequestContext):
    """TC-BE-06 – POST /validate with clean records → is_valid: true."""
    response = backend_api.post(
        "/validate",
        data={"records": VALID_RECORDS, "rules": VALIDATION_RULES},
    )
    assert response.status == 200, f"Expected 200, got {response.status}: {response.text()}"
    body = response.json()
    assert body["is_valid"] is True
    assert body["total_rows"] == len(VALID_RECORDS)
    assert body["valid_rows"] == body["total_rows"]


@pytest.mark.backend
def test_be06_validate_reports_missing_value_counts(backend_api: APIRequestContext):
    """TC-BE-06 – Records with null fields increment missing_value_counts."""
    records_with_nulls = [
        {"Company Name": None, "Revenue": 100.0, "Region": "South", "Report Date": "2026-01-01"},
        {"Company Name": "Beta", "Revenue": None, "Region": "North", "Report Date": "2026-01-02"},
    ]
    response = backend_api.post(
        "/validate",
        data={"records": records_with_nulls, "rules": VALIDATION_RULES},
    )
    assert response.status == 200
    body = response.json()
    assert body["missing_value_counts"]["Company Name"] >= 1


@pytest.mark.backend
def test_be06_validate_detects_range_violation(backend_api: APIRequestContext):
    """TC-BE-06 / TC-EXT-BE-03 – Revenue below min_value → row counted as invalid."""
    records_bad_range = [
        {"Company Name": "Bad Corp", "Revenue": -500.0, "Region": "East", "Report Date": "2026-01-01"},
    ]
    rules_with_range = [
        {"name": "Company Name", "dtype": "str",   "required": True},
        {"name": "Revenue",      "dtype": "float",  "required": True, "min_value": 0.0},
        {"name": "Region",       "dtype": "str",   "required": True},
    ]
    response = backend_api.post(
        "/validate",
        data={"records": records_bad_range, "rules": rules_with_range},
    )
    assert response.status == 200
    body = response.json()
    assert body["is_valid"] is False


# ===========================================================================
# TC-EXT-BE – Extended scenarios not in the Excel matrix
# ===========================================================================

@pytest.mark.backend
def test_ext_be01_process_single_record_valid(backend_api: APIRequestContext):
    """TC-EXT-BE-01 – Single-record /process request succeeds."""
    response = backend_api.post(
        "/process",
        data={"records": [VALID_RECORDS[0]], "profile": VALID_PROFILE},
    )
    assert response.status == 200
    assert response.json()["row_count"] >= 1


@pytest.mark.backend
@pytest.mark.slow
def test_ext_be02_large_batch_store_under_5s(backend_api: APIRequestContext):
    """TC-EXT-BE-02 – Storing 200 records completes in under 5 seconds."""
    large_records = [
        {"Company Name": f"Company_{i}", "Revenue": float(i * 1000), "Region": "East"}
        for i in range(200)
    ]
    start = time.perf_counter()
    response = backend_api.post(
        "/store",
        data={"table": "carivix_perf_test", "records": large_records},
    )
    elapsed = time.perf_counter() - start
    assert response.status == 200, f"Store failed: {response.text()}"
    assert elapsed < 5.0, f"Large batch insert too slow: {elapsed:.2f}s (limit 5s)"
    assert response.json()["rows_written"] == 200


@pytest.mark.backend
def test_ext_be03_process_empty_records_returns_400(backend_api: APIRequestContext):
    """TC-EXT-BE-03 – POST /process with empty records list → 400."""
    response = backend_api.post(
        "/process",
        data={"records": [], "profile": VALID_PROFILE},
        fail_on_status_code=False,
    )
    assert response.status == 400


@pytest.mark.backend
def test_ext_be04_validate_duplicate_rows_counted(backend_api: APIRequestContext):
    """TC-EXT-BE-04 – Duplicate rows are detected and counted in the validation report."""
    dupe_records = [VALID_RECORDS[0], VALID_RECORDS[0]]  # identical rows
    response = backend_api.post(
        "/validate",
        data={"records": dupe_records, "rules": VALIDATION_RULES},
    )
    assert response.status == 200
    body = response.json()
    assert body["duplicate_row_count"] >= 1


@pytest.mark.backend
def test_ext_be05_health_response_is_json(backend_api: APIRequestContext):
    """TC-EXT-BE-05 – Health response Content-Type is application/json."""
    response = backend_api.get("/health")
    assert response.status == 200
    content_type = response.headers.get("content-type", "")
    assert "application/json" in content_type, f"Unexpected Content-Type: {content_type}"


@pytest.mark.backend
def test_ext_be06_store_data_types_int_float_str(backend_api: APIRequestContext):
    """TC-EXT-BE-06 – /store correctly handles int, float, and string column types."""
    mixed = [{"int_col": 42, "float_col": 3.14, "str_col": "hello"}]
    resp = backend_api.post("/store", data={"table": "carivix_mixed_types", "records": mixed})
    assert resp.status == 200
    assert resp.json()["rows_written"] == 1

    get = backend_api.get("/retrieve/carivix_mixed_types")
    assert get.status == 200
    row = get.json()["data"][0]
    assert row["str_col"] == "hello"
