"""
Standalone End-to-End Workflow Execution Test
===============================================
Demonstrates a complete, multi-stage enterprise data workflow:
  1. Ingestion: Discover CSV data sources via GET /sources
  2. Transformation: ETL processing & date decomposition via POST /process
  3. Quality Assurance: Schema & type validation via POST /validate
  4. Persistence: Database table creation & transactional insert via POST /store
  5. Retrieval & Verification: Stored data query & integrity check via GET /retrieve/{table}
"""

import sys
import time
import pytest
from playwright.sync_api import APIRequestContext

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@pytest.mark.backend
def test_complete_end_to_end_workflow(backend_api: APIRequestContext):
    print("\n" + "=" * 70)
    print("🚀 EXECUTING COMPLETE END-TO-END ENTERPRISE WORKFLOW")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Service Health & Readiness
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    res_health = backend_api.get("/health")
    assert res_health.status == 200
    health_data = res_health.json()
    assert health_data.get("status") == "ok"
    lat_health = (time.perf_counter() - t0) * 1000
    print(f"✅ Step 1 [Health Check]      : HTTP 200 OK | Status: {health_data['status']} ({lat_health:.1f} ms)")

    # -----------------------------------------------------------------------
    # Step 2: Data Acquisition & Source Cataloging
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    res_sources = backend_api.get("/sources")
    assert res_sources.status == 200
    sources_data = res_sources.json()
    csv_sources = sources_data.get("csv_sources", [])
    assert len(csv_sources) > 0
    lat_sources = (time.perf_counter() - t0) * 1000
    print(f"✅ Step 2 [Source Discovery]  : HTTP 200 OK | Found {len(csv_sources)} registered CSV sources ({lat_sources:.1f} ms)")

    # -----------------------------------------------------------------------
    # Step 3: ETL Processing & Normalization
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    raw_records = [
        {"Company Name": "CARIVIX Tech Global", "Revenue": 1_250_000.0, "Region": "South", "Report Date": "2026-04-01"},
        {"Company Name": "Alpha Analytics Corp", "Revenue": 890_000.0, "Region": "North", "Report Date": "2026-04-02"},
        {"Company Name": "Apex Systems Inc", "Revenue": 3_100_000.0, "Region": "East", "Report Date": "2026-04-03"},
    ]
    res_process = backend_api.post(
        "/process",
        data={"records": raw_records, "profile": "company_financials"},
    )
    assert res_process.status == 200
    processed_data = res_process.json()
    assert processed_data.get("row_count") == 3
    normalized_rows = processed_data.get("data", [])
    assert len(normalized_rows) == 3
    # Check temporal feature engineering
    assert "report_date_year" in normalized_rows[0]
    assert "report_date_quarter" in normalized_rows[0]
    lat_process = (time.perf_counter() - t0) * 1000
    print(f"✅ Step 3 [ETL Transformation]: HTTP 200 OK | Ingested {len(raw_records)} rows -> Output {len(normalized_rows)} normalized rows ({lat_process:.1f} ms)")

    # -----------------------------------------------------------------------
    # Step 4: Data Quality & Schema Validation
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    rules = [
        {"name": "company_name", "required": True},
        {"name": "revenue", "dtype": "float64", "required": True},
        {"name": "report_date", "required": True},
    ]
    res_val = backend_api.post(
        "/validate",
        data={"records": normalized_rows, "rules": rules},
    )
    assert res_val.status == 200
    val_data = res_val.json()
    assert val_data.get("is_valid") is True
    assert len(val_data.get("errors", [])) == 0
    lat_val = (time.perf_counter() - t0) * 1000
    print(f"✅ Step 4 [Quality Validation]: HTTP 200 OK | Conformance: Valid, 0 Errors ({lat_val:.1f} ms)")

    # -----------------------------------------------------------------------
    # Step 5: Database Persistence Layer
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    table_name = f"e2e_verified_workflow_{int(time.time())}"
    res_store = backend_api.post(
        "/store",
        data={"table": table_name, "records": normalized_rows},
    )
    assert res_store.status == 200
    store_data = res_store.json()
    assert store_data.get("rows_written") == 3
    lat_store = (time.perf_counter() - t0) * 1000
    print(f"✅ Step 5 [Database Storage]  : HTTP 200 OK | Written 3 records into table '{table_name}' ({lat_store:.1f} ms)")

    # -----------------------------------------------------------------------
    # Step 6: Retrieval & Round-Trip Data Integrity Verification
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    res_retrieve = backend_api.get(f"/retrieve/{table_name}")
    assert res_retrieve.status == 200
    retrieved_data = res_retrieve.json()
    retrieved_records = retrieved_data.get("data", [])
    assert len(retrieved_records) == 3
    retrieved_names = [r["company_name"] for r in retrieved_records]
    assert "CARIVIX Tech Global" in retrieved_names
    lat_retrieve = (time.perf_counter() - t0) * 1000
    print(f"✅ Step 6 [Data Verification] : HTTP 200 OK | Retrieved 3 rows matching input data ({lat_retrieve:.1f} ms)")

    print("=" * 70)
    print("🎉 END-TO-END WORKFLOW VERIFIED (6/6 STAGES PASSED)")
    print("=" * 70 + "\n")
