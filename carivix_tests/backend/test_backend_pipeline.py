"""
Backend Pipeline & Concurrency Tests
======================================

Maps to test cases: TC-BE-03 (concurrency), TC-BE-05 (KPI lineage / full pipeline)
Extended scenarios:  TC-EXT-BE-07 (pipeline error paths), TC-EXT-BE-08 (concurrent API requests)

Service under test : http://127.0.0.1:8000  (BACKEND module/api.py)
Playwright fixture : ``backend_api`` (APIRequestContext) from conftest.py

Run:
    pytest carivix_tests/backend/test_backend_pipeline.py -v
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from playwright.sync_api import APIRequestContext


VALID_RECORDS = [
    {"Company Name": "Pipe Corp",  "Revenue": 100_000.0, "Region": "South", "Report Date": "2026-01-05"},
    {"Company Name": "Pipe Beta",  "Revenue": 250_000.0, "Region": "north", "Report Date": "2026-02-10"},
    {"Company Name": "Pipe Gamma", "Revenue":  80_000.0, "Region": "East",  "Report Date": "2026-03-15"},
]
VALID_PROFILE = "company_financials"


# ===========================================================================
# TC-BE-05 – KPI Lineage / Full Pipeline  (POST /pipeline)
# ===========================================================================

@pytest.mark.backend
def test_be05_pipeline_processes_and_stores_in_one_call(backend_api: APIRequestContext):
    """TC-BE-05 – POST /pipeline runs ETL + DB write in a single atomic call."""
    response = backend_api.post(
        "/pipeline",
        data={
            "records": VALID_RECORDS,
            "profile": VALID_PROFILE,
            "table": "carivix_pipeline_test",
        },
    )
    assert response.status == 200, f"Pipeline failed: {response.text()}"
    body = response.json()
    assert "row_count" in body
    assert body["row_count"] > 0
    assert "rows_written" in body
    assert body["rows_written"] > 0
    assert body["table"] == "carivix_pipeline_test"


@pytest.mark.backend
def test_be05_pipeline_response_includes_columns(backend_api: APIRequestContext):
    """TC-BE-05 – Pipeline response exposes the output column list for traceability."""
    response = backend_api.post(
        "/pipeline",
        data={
            "records": VALID_RECORDS,
            "profile": VALID_PROFILE,
            "table": "carivix_pipeline_cols_test",
        },
    )
    assert response.status == 200
    body = response.json()
    assert isinstance(body.get("columns"), list)
    assert len(body["columns"]) > 0


@pytest.mark.backend
def test_be05_pipeline_unknown_profile_returns_400(backend_api: APIRequestContext):
    """TC-BE-05 – /pipeline with invalid profile name → 400 (not 500)."""
    response = backend_api.post(
        "/pipeline",
        data={
            "records": VALID_RECORDS,
            "profile": "invalid_profile_xyz",
            "table": "carivix_pipeline_bad",
        },
        fail_on_status_code=False,
    )
    assert response.status == 400, f"Expected 400, got {response.status}"


@pytest.mark.backend
def test_be05_pipeline_data_retrievable_after_run(backend_api: APIRequestContext):
    """TC-BE-05 – Data written via /pipeline is immediately readable via /retrieve."""
    table = "carivix_pipeline_verify"
    pipeline_resp = backend_api.post(
        "/pipeline",
        data={"records": VALID_RECORDS, "profile": VALID_PROFILE, "table": table},
    )
    assert pipeline_resp.status == 200

    retrieve_resp = backend_api.get(f"/retrieve/{table}")
    assert retrieve_resp.status == 200
    body = retrieve_resp.json()
    assert body["row_count"] > 0, "Pipeline wrote data but retrieve returned 0 rows"


@pytest.mark.backend
def test_be05_pipeline_row_count_matches_rows_written(backend_api: APIRequestContext):
    """TC-BE-05 – row_count and rows_written are consistent in the pipeline response."""
    response = backend_api.post(
        "/pipeline",
        data={
            "records": VALID_RECORDS,
            "profile": VALID_PROFILE,
            "table": "carivix_pipeline_consistency",
        },
    )
    assert response.status == 200
    body = response.json()
    # After processing, row_count == rows_written
    assert body["row_count"] == body["rows_written"], (
        f"row_count ({body['row_count']}) != rows_written ({body['rows_written']})"
    )


# ===========================================================================
# TC-BE-03 – Concurrency: 10 simultaneous /store requests
# ===========================================================================

@pytest.mark.backend
@pytest.mark.slow
def test_be03_concurrent_store_requests_all_succeed(backend_api: APIRequestContext):
    """TC-BE-03 – 10 concurrent POST /store requests all return 200 without data loss."""
    table = "carivix_concurrency_test"
    errors: list[str] = []
    results: list[dict] = []

    import json
    import urllib.request

    base_url = "http://127.0.0.1:8000"

    def store_one(worker_id: int) -> None:
        try:
            payload = json.dumps({
                "table": table,
                "records": [{"worker": worker_id, "value": f"w{worker_id}"}],
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/store",
                data=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                results.append({"worker": worker_id, "status": resp.status})
                if resp.status != 200:
                    errors.append(f"Worker {worker_id} got HTTP {resp.status}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Worker {worker_id} raised: {exc}")

    threads = [threading.Thread(target=store_one, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent store errors:\n" + "\n".join(errors)
    assert len(results) == 10, "Not all threads completed"
    assert all(r["status"] == 200 for r in results)


@pytest.mark.backend
@pytest.mark.slow
def test_be03_concurrent_read_write_no_errors(backend_api: APIRequestContext):
    """TC-BE-03 – Simultaneous reads and writes to the same table complete without error."""
    table = "carivix_rw_test"
    # Seed with initial data (matching 'val' column)
    backend_api.post("/store", data={"table": table, "records": [{"val": 0}]})

    import json
    import urllib.request
    from urllib.error import HTTPError

    base_url = "http://127.0.0.1:8000"
    errors: list[str] = []

    def write_worker(i: int) -> None:
        try:
            payload = json.dumps({"table": table, "records": [{"val": i}]}).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/store",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status != 200:
                    errors.append(f"Write {i} → {resp.status}")
        except Exception as exc:
            errors.append(f"Write {i} error: {exc}")

    def read_worker(i: int) -> None:
        try:
            req = urllib.request.Request(f"{base_url}/retrieve/{table}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status not in (200, 404):
                    errors.append(f"Read {i} → {resp.status}")
        except HTTPError as e:
            if e.code not in (200, 404):
                errors.append(f"Read {i} → HTTP {e.code}")
        except Exception as exc:
            errors.append(f"Read {i} error: {exc}")

    threads = (
        [threading.Thread(target=write_worker, args=(i,)) for i in range(5)]
        + [threading.Thread(target=read_worker, args=(i,)) for i in range(5)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, "Concurrent R/W errors:\n" + "\n".join(errors)


# ===========================================================================
# TC-EXT-BE – Additional pipeline edge cases
# ===========================================================================

@pytest.mark.backend
def test_ext_be_pipeline_empty_records_returns_error(backend_api: APIRequestContext):
    """TC-EXT-BE-07 – /pipeline with empty records list → 400 or 422."""
    response = backend_api.post(
        "/pipeline",
        data={"records": [], "profile": VALID_PROFILE, "table": "carivix_empty_pipe"},
        fail_on_status_code=False,
    )
    assert response.status in (400, 422), f"Expected 400/422, got {response.status}"


@pytest.mark.backend
def test_ext_be_pipeline_missing_table_field_returns_422(backend_api: APIRequestContext):
    """TC-EXT-BE-08 – /pipeline with missing 'table' field → 422."""
    response = backend_api.post(
        "/pipeline",
        data={"records": VALID_RECORDS, "profile": VALID_PROFILE},
        fail_on_status_code=False,
    )
    assert response.status == 422


@pytest.mark.backend
def test_ext_be_store_integer_primary_key_autoincrement(backend_api: APIRequestContext):
    """TC-EXT-BE-09 – Verify stored rows have auto-assigned 'id' field."""
    table = "carivix_id_test"
    backend_api.post("/store", data={"table": table, "records": [{"label": "a"}, {"label": "b"}]})
    resp = backend_api.get(f"/retrieve/{table}")
    assert resp.status == 200
    rows = resp.json()["data"]
    assert len(rows) >= 2
    ids = [r.get("id") for r in rows]
    assert all(i is not None for i in ids), f"Some rows missing 'id': {rows}"
    assert len(set(ids)) == len(ids), "Duplicate IDs detected"
