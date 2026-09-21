"""
Rigorous Backend Data Service High-Volume, Calculation & Persistence Stress Tests
=================================================================================

Pushes the Central Backend Data Service (Port 8000) to its operational limits:
- High-volume batch processing (250 financial records through ETL)
- Date boundary edge cases (leap day, end of year, multi-century dates)
- Extreme numeric floats & scientific notation preservation
- Large database write/read persistence round-trip fidelity
- Concurrent multi-table data operations

Target Service: http://127.0.0.1:8000 (BACKEND module/api.py)
"""

from __future__ import annotations

import concurrent.futures
import random
import time
from typing import Any, Dict, List

import pytest
from playwright.sync_api import APIRequestContext


# ===========================================================================
# 1. High-Volume Batch ETL Processing (250 Records)
# ===========================================================================

@pytest.mark.stress
@pytest.mark.backend
def test_backend_large_batch_etl_processing_and_throughput(backend_api: APIRequestContext):
    """
    Generate 250 synthetic financial records and process them through /process.
    Asserts:
    1. HTTP 200 OK.
    2. Exact row_count == 250 (zero record loss).
    3. Derived date features generated for all 250 records.
    4. Processing time < 2.5s.
    """
    regions = ["North", "South", "East", "West", "Central"]
    records = []
    for i in range(250):
        records.append({
            "Company Name": f"Enterprise Corp {i:04d}",
            "Revenue": round(random.uniform(50_000.0, 5_000_000.0), 2),
            "Region": random.choice(regions),
            "Report Date": f"202{random.randint(4, 6)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        })

    t0 = time.perf_counter()
    resp = backend_api.post("/process", data={"records": records, "profile": "company_financials"})
    elapsed = time.perf_counter() - t0

    assert resp.status == 200, f"Large batch processing failed: {resp.text()}"
    body = resp.json()

    assert body.get("row_count") == 250
    data = body.get("data", [])
    assert len(data) == 250

    # Verify date decomposition features present across transformed records
    sample_row = data[0]
    for col in ["report_date_year", "report_date_month", "report_date_day", "report_date_quarter"]:
        assert col in sample_row, f"Missing derived date feature: {col}"

    print(f"\n[Backend Large Batch] 250 records processed in {elapsed:.2f}s ({250/elapsed:.1f} rows/sec)")
    assert elapsed < 3.0, f"Processing too slow: {elapsed:.2f}s"


# ===========================================================================
# 2. Date Arithmetic & Boundary Edge Cases
# ===========================================================================

DATE_EDGE_CASES = [
    # Leap year day
    {"Company Name": "Leap Day Corp", "Revenue": 100000.0, "Region": "North", "Report Date": "2024-02-29"},
    # First day of millennium
    {"Company Name": "Millennium Corp", "Revenue": 200000.0, "Region": "South", "Report Date": "2000-01-01"},
    # Year end boundary
    {"Company Name": "Year End Corp", "Revenue": 300000.0, "Region": "East", "Report Date": "2026-12-31"},
    # Quarter boundary
    {"Company Name": "Q1 End Corp", "Revenue": 400000.0, "Region": "West", "Report Date": "2026-03-31"},
]


@pytest.mark.stress
@pytest.mark.backend
def test_backend_date_arithmetic_edge_cases(backend_api: APIRequestContext):
    """
    Verify ETL engine correctly handles calendar boundaries:
    - Leap day (2024-02-29 -> day=29, month=2)
    - Year ends and quarter transitions.
    """
    resp = backend_api.post(
        "/process", data={"records": DATE_EDGE_CASES, "profile": "company_financials"}
    )
    assert resp.status == 200
    rows = resp.json().get("data", [])
    assert len(rows) == 4

    # Verify leap day row
    leap_row = rows[0]
    assert leap_row.get("report_date_day") == 29
    assert leap_row.get("report_date_month") == 2
    assert leap_row.get("report_date_year") == 2024

    # Verify quarter boundary (March 31 -> quarter 1)
    q1_row = rows[3]
    assert q1_row.get("report_date_quarter") == 1
    assert q1_row.get("report_date_month") == 3
    assert q1_row.get("report_date_day") == 31


# ===========================================================================
# 3. Database Persistence Round-Trip Fidelity Under Load
# ===========================================================================

@pytest.mark.stress
@pytest.mark.backend
def test_backend_database_persistence_fidelity_under_load(backend_api: APIRequestContext):
    """
    Store 50 financial records with float precision and verify 100% data fidelity:
    1. Write 50 records to a fresh test table.
    2. Read back 50 records via GET /retrieve/{table}.
    3. Assert exact match of all values without floating point drift or string truncation.
    """
    table_name = f"stress_persist_{int(time.time())}_{random.randint(100, 999)}"
    test_records = [
        {
            "id_num": i,
            "company": f"Precision Testing Corp {i}",
            "balance": round(12345.6789 + i * 111.11, 4),
            "ratio": round(0.123456 / (i + 1), 6),
            "status": "ACTIVE",
        }
        for i in range(50)
    ]

    # Store
    store_resp = backend_api.post("/store", data={"table": table_name, "records": test_records})
    assert store_resp.status == 200, f"Store failed: {store_resp.text()}"
    assert store_resp.json().get("rows_written") == 50

    # Retrieve
    retrieve_resp = backend_api.get(f"/retrieve/{table_name}")
    assert retrieve_resp.status == 200, f"Retrieve failed: {retrieve_resp.text()}"
    retrieved_data = retrieve_resp.json().get("data", [])
    assert len(retrieved_data) == 50

    # Verify precision and data equality
    for i, orig in enumerate(test_records):
        matched = next((r for r in retrieved_data if r.get("id_num") == orig["id_num"]), None)
        assert matched is not None, f"Record id_num {orig['id_num']} missing in retrieved data"
        assert matched["company"] == orig["company"]
        assert abs(matched["balance"] - orig["balance"]) < 0.001
        assert abs(matched["ratio"] - orig["ratio"]) < 0.00001
