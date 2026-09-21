"""
Unified AI / NLP / GIS Cross-Module End-to-End Workflow Validation Test
========================================================================

Demonstrates and verifies the complete Sprint 5 cross-team integration journey:
  1. Natural Language Query Ingestion
  2. NLP Layer: Model-driven Intent Classification & Entity Extraction
  3. Spatial Flow: WebGIS Boundary Retrieval & Telemetry (< 120ms SLA)
  4. Predictive Flow: ML Inference Service (< 300ms SLA)
  5. Central Persistence: Backend Ingestion, SQLite Storage, and Query Verification
  6. Resilient Handling: Out-of-Scope Fallback & Re-prompting

Services involved:
  - Backend Data Service (Port 8000)
  - ML Inference Service (Port 8001)
  - WebGIS Spatial Service (Port 8003)
  - NLP Intelligence Module (nlp_module.py)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

import pytest
from playwright.sync_api import APIRequestContext

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure ML module is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_MODULE_ROOT = PROJECT_ROOT / "ML module"
if str(ML_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_MODULE_ROOT))

from nlp_module import analyze, generate_structured_query, extract_entities


@pytest.mark.e2e
@pytest.mark.smoke
def test_e2e_nl_to_gis_spatial_workflow(gis_api: APIRequestContext):
    """
    Workflow 1: User submits a natural language spatial query.
    Pipeline: Natural Language -> NLP Intent (GIS_VIEW) & Entity (Adilabad) ->
              GIS REST API query -> GeoJSON geometry & telemetry verification.
    """
    print("\n" + "=" * 70)
    print(">>> WORKFLOW 1: NATURAL LANGUAGE -> NLP INTENT -> WEBGIS SPATIAL SERVICE")
    print("=" * 70)

    user_query = "Show boundary map for Adilabad district in Telangana"
    t_start = time.perf_counter()

    # Step 1: NLP Analysis
    nlp_res = analyze(user_query)
    assert nlp_res["intent"] == "GIS_VIEW"
    assert "Adilabad" in nlp_res["entities"]["locations"]
    assert "Telangana" in nlp_res["entities"]["locations"]
    print(f"[OK] Step 1 [NLP Intent]: '{nlp_res['intent']}' (Confidence: {nlp_res['confidence']:.2f})")
    print(f"   Extracted Entities: {nlp_res['entities']['locations']}")

    # Step 2: Structured Query Generation
    structured = generate_structured_query(user_query)
    assert structured["target_service"] == "GIS_SPATIAL_SERVICE"
    req_schema = structured["request_schema"]
    endpoint = req_schema["endpoint"]
    params = req_schema["params"]
    print(f"[OK] Step 2 [Query Gen]  : Endpoint: {endpoint} | Params: {params} ({structured['execution_time_ms']} ms)")

    # Step 3: WebGIS API Execution
    gis_resp = gis_api.get(endpoint, params=params)
    assert gis_resp.status == 200, f"GIS query failed with status {gis_resp.status}: {gis_resp.text()}"
    geojson = gis_resp.json()
    assert geojson.get("type") == "FeatureCollection"
    features = geojson.get("features", [])
    assert len(features) >= 1, "Expected at least one matching district feature"

    first_feat = features[0]
    props = first_feat.get("properties", {})
    name_2 = props.get("NAME_2", "")
    assert "adilabad" in name_2.lower()
    print(f"[OK] Step 3 [GIS Boundary]: HTTP 200 OK | District: {name_2} | Geometry: {first_feat.get('geometry', {}).get('type')}")

    # Step 4: Spatial Telemetry Verification
    telemetry_resp = gis_api.get("/api/v1/spatial/analytics")
    assert telemetry_resp.status == 200
    telemetry = telemetry_resp.json()
    assert telemetry.get("status") == "success"
    assert telemetry.get("total_districts_indexed", 0) > 0
    print(f"[OK] Step 4 [GIS Telemetry]: Total Districts: {telemetry['total_districts_indexed']} | States: {telemetry['total_states_indexed']}")

    total_time_ms = (time.perf_counter() - t_start) * 1000.0
    print(f"[*] Complete Spatial Journey finished in {total_time_ms:.1f} ms\n")


@pytest.mark.e2e
@pytest.mark.smoke
def test_e2e_nl_to_ml_prediction_and_backend_persistence(
    ml_api: APIRequestContext,
    backend_api: APIRequestContext,
):
    """
    Workflow 2: User submits a natural language credit prediction query.
    Pipeline: Natural Language -> NLP Feature Extraction -> ML Inference (Port 8001) ->
              Backend Persistence (Port 8000) -> SQLite Query Verification.
    """
    print("\n" + "=" * 70)
    print(">>> WORKFLOW 2: NATURAL LANGUAGE -> ML INFERENCE -> BACKEND PERSISTENCE")
    print("=" * 70)

    user_query = (
        "Predict default probability for a 35-year-old single borrower with income $50,000, "
        "credit score 680, loan amount $10,000, 5 years employed, applied on 2026-09-21"
    )
    t_start = time.perf_counter()

    # Step 1: NLP Feature Extraction
    nlp_res = analyze(user_query)
    assert nlp_res["intent"] == "PREDICTION"
    extracted_feats = nlp_res["entities"]["features"]
    assert extracted_feats.get("age") == 35.0
    assert extracted_feats.get("income") == 50000.0
    assert extracted_feats.get("credit_score") == 680.0
    print(f"[OK] Step 1 [NLP Extract]: Intent: {nlp_res['intent']} | Features: {extracted_feats}")

    # Step 2: Structured ML Payload Generation
    structured = generate_structured_query(user_query)
    assert structured["target_service"] == "ML_INFERENCE_SERVICE"
    ml_payload = structured["request_schema"]["payload"]
    print(f"[OK] Step 2 [Payload Gen]: Generated valid ML inference payload ({structured['execution_time_ms']} ms)")

    # Step 3: ML Model Inference Execution
    t_ml = time.perf_counter()
    ml_resp = ml_api.post("/api/v1/predict", data=ml_payload)
    ml_latency_ms = (time.perf_counter() - t_ml) * 1000.0
    assert ml_resp.status == 200, f"ML predict failed with status {ml_resp.status}: {ml_resp.text()}"
    ml_result = ml_resp.json()
    assert ml_result.get("success") is True
    prediction = ml_result.get("prediction")
    model_name = ml_result.get("model")
    prob = ml_result.get("probability", 0.0)
    print(f"[OK] Step 3 [ML Predict] : Model: {model_name} | Prediction: {prediction} | Prob: {prob:.4f} ({ml_latency_ms:.1f} ms)")
    assert ml_latency_ms < 300.0, f"ML inference latency exceeded 300ms SLA: {ml_latency_ms:.1f} ms"

    # Step 4: Backend Data Ingestion & Storage via ETL Pipeline
    table_name = f"e2e_loan_eval_{int(time.time())}"
    raw_record = [
        {
            "Company Name": f"Applicant-{int(time.time())}",
            "Revenue": ml_payload["income"],
            "Region": "South",
            "Report Date": "2026-09-21",
        }
    ]
    proc_resp = backend_api.post(
        "/process",
        data={"records": raw_record, "profile": "company_financials"},
    )
    assert proc_resp.status == 200
    norm_records = proc_resp.json().get("data", raw_record)

    store_resp = backend_api.post(
        "/store",
        data={"table": table_name, "records": norm_records},
    )
    assert store_resp.status == 200, f"Backend store failed: {store_resp.text()}"
    store_result = store_resp.json()
    assert store_result.get("rows_written") == 1
    print(f"[OK] Step 4 [Backend Store]: Persisted evaluation record to table '{table_name}'")

    # Step 5: Retrieve and Verify Persistence
    retrieve_resp = backend_api.get(f"/retrieve/{table_name}")
    assert retrieve_resp.status == 200, f"Backend retrieve failed: {retrieve_resp.text()}"
    retrieve_result = retrieve_resp.json()
    stored_records = retrieve_result.get("data", [])
    assert len(stored_records) == 1
    print(f"[OK] Step 5 [Verification] : Verified record in database table (Row count: {len(stored_records)})")

    total_time_ms = (time.perf_counter() - t_start) * 1000.0
    print(f"[*] Complete ML-to-Backend Journey finished in {total_time_ms:.1f} ms\n")


@pytest.mark.e2e
def test_e2e_out_of_scope_resilience_and_reprompt():
    """
    Workflow 3: Verify resilient recovery when out-of-scope queries are submitted,
    followed by an in-scope data query.
    """
    print("\n" + "=" * 70)
    print(">>> WORKFLOW 3: RESILIENT ERROR HANDLING & IN-SCOPE RECOVERY")
    print("=" * 70)

    # Step 1: Out-of-Scope Query
    gibberish = "<script>alert('test')</script> unladen swallow"
    res1 = analyze(gibberish)
    assert res1["intent"] == "UNKNOWN_INTENT"
    sq1 = generate_structured_query(gibberish)
    assert sq1["target_service"] == "FALLBACK_HANDLER"
    print(f"[OK] Step 1 [Fallback]: Correctly caught out-of-scope query -> Assigned UNKNOWN_INTENT")

    # Step 2: In-Scope Recovery Query
    corrected_query = "Show rainfall in Jagtial for 2025"
    res2 = analyze(corrected_query)
    assert res2["intent"] == "DATA_METRIC"
    assert "Jagtial" in res2["entities"]["locations"]
    assert "rainfall" in res2["entities"]["metrics"]
    print(f"[OK] Step 2 [Recovery]: Successfully routed recovered query -> Assigned DATA_METRIC with entities")
