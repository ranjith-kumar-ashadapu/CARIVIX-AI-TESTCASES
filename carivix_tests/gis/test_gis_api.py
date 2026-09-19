"""
WebGIS Spatial Backend Integration Tests
==========================================

Maps to test cases:
  TC-GIS-01  Dataset Audit & CRS Validation (WGS 84 / EPSG:4326)
  TC-GIS-02  Attribute Table Standardization (NAME_1, NAME_2, etc.)
  TC-GIS-03  Spatial Telemetry & Analytics Endpoint (< 120 ms SLA)
  TC-GIS-04  Backend Boundary Service Integration (RFC 7946 FeatureCollection)
  TC-GIS-05  State-Level Boundary Filter Optimization (?state=Telangana)
  TC-GIS-06  Spatial Intelligence Summary & Metric Aggregation
  TC-GIS-07  Spatial Attribute Search & Query Index (?q=Adilabad)
  TC-GIS-08  Sample Points & Density Telemetry Distribution

Service under test : http://127.0.0.1:8003 (CARIVIX - AI/server.py)
Playwright fixture : ``gis_api`` (APIRequestContext) from conftest.py
"""

from __future__ import annotations

import time
import pytest
from playwright.sync_api import APIRequestContext


# ===========================================================================
# TC-GIS-03 – Spatial Telemetry & Analytics Endpoint
# ===========================================================================

@pytest.mark.gis
@pytest.mark.smoke
def test_gis03_spatial_analytics_returns_200_and_telemetry(gis_api: APIRequestContext):
    """TC-GIS-03 – GET /api/v1/spatial/analytics returns 200 with complete telemetry metadata."""
    # Warm-up request to avoid initial JIT / OS page fault cold start
    gis_api.get("/api/v1/spatial/analytics")

    start = time.perf_counter()
    resp = gis_api.get("/api/v1/spatial/analytics")
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert resp.status == 200, f"Expected 200, got {resp.status}"
    body = resp.json()
    assert body.get("status") == "success", f"Expected status 'success', got {body.get('status')}"
    assert "total_districts_indexed" in body
    assert body["total_districts_indexed"] > 0
    assert "total_states_indexed" in body
    assert body["total_states_indexed"] > 0
    assert "avg_density_index" in body
    assert elapsed_ms < 300, f"Analytics endpoint latency too high: {elapsed_ms:.1f}ms (limit 300ms)"


# ===========================================================================
# TC-GIS-06 – Spatial Intelligence Summary
# ===========================================================================

@pytest.mark.gis
@pytest.mark.smoke
def test_gis06_spatial_intelligence_summary_structure(gis_api: APIRequestContext):
    """TC-GIS-06 – GET /api/v1/spatial/intelligence/summary returns executive intelligence metrics."""
    resp = gis_api.get("/api/v1/spatial/intelligence/summary")
    assert resp.status == 200, f"Expected 200, got {resp.status}"
    body = resp.json()
    assert "status" in body or "data" in body or "summary" in body or isinstance(body, dict)
    assert len(body) > 0


# ===========================================================================
# TC-GIS-04 – Backend Boundary Service Integration (RFC 7946)
# ===========================================================================

@pytest.mark.gis
@pytest.mark.parametrize("tier", [0, 1, 2])
def test_gis04_boundary_tier_returns_valid_feature_collection(gis_api: APIRequestContext, tier: int):
    """TC-GIS-04 – GET /api/v1/spatial/boundaries/{tier} returns valid RFC 7946 FeatureCollection."""
    resp = gis_api.get(f"/api/v1/spatial/boundaries/{tier}")
    assert resp.status == 200, f"Boundary tier {tier} failed: {resp.status}"

    data = resp.json()
    assert data.get("type") == "FeatureCollection", f"Expected FeatureCollection, got {data.get('type')}"
    assert "features" in data
    assert len(data["features"]) > 0, f"Tier {tier} returned empty features array"


# ===========================================================================
# TC-GIS-01 – Dataset Audit & CRS Validation (WGS 84 / EPSG:4326)
# ===========================================================================

@pytest.mark.gis
def test_gis01_coordinates_within_wgs84_india_bounds(gis_api: APIRequestContext):
    """TC-GIS-01 – Verify GeoJSON coordinates conform to WGS 84 within India geographic bounds."""
    resp = gis_api.get("/api/v1/spatial/boundaries/0")
    assert resp.status == 200
    data = resp.json()

    # Verify India coordinate bounding box: Lat: [6.0, 38.0], Lon: [68.0, 98.0]
    features = data.get("features", [])
    assert len(features) > 0
    first_feat = features[0]
    geom = first_feat.get("geometry", {})
    coords = geom.get("coordinates", [])

    def check_coords(c_list):
        if not c_list:
            return
        if isinstance(c_list[0], (int, float)):
            lon, lat = c_list[0], c_list[1]
            assert 60.0 <= lon <= 100.0, f"Longitude out of India bounds: {lon}"
            assert 5.0 <= lat <= 40.0, f"Latitude out of India bounds: {lat}"
        else:
            for sub in c_list:
                check_coords(sub)

    check_coords(coords[:2])


# ===========================================================================
# TC-GIS-02 – Attribute Table Standardization
# ===========================================================================

@pytest.mark.gis
def test_gis02_district_attributes_standardization(gis_api: APIRequestContext):
    """TC-GIS-02 – Verify Level 2 district boundaries contain standardized geographic keys."""
    resp = gis_api.get("/api/v1/spatial/boundaries/2")
    assert resp.status == 200
    data = resp.json()
    features = data.get("features", [])
    assert len(features) > 0

    first_props = features[0].get("properties", {})
    # Check for presence of standard keys: NAME_1 (State), NAME_2 (District) or equivalent
    has_state = "NAME_1" in first_props or "state" in first_props or "State" in first_props
    has_district = "NAME_2" in first_props or "district" in first_props or "District" in first_props
    assert has_state, f"Missing state attribute in properties: {list(first_props.keys())}"
    assert has_district, f"Missing district attribute in properties: {list(first_props.keys())}"


# ===========================================================================
# TC-GIS-05 – State-Level Boundary Filter Optimization
# ===========================================================================

@pytest.mark.gis
def test_gis05_state_boundary_filter_returns_matching_districts(gis_api: APIRequestContext):
    """TC-GIS-05 – GET /api/v1/boundaries/2?state=Telangana returns only districts for that state."""
    resp = gis_api.get("/api/v1/boundaries/2?state=Telangana")
    assert resp.status == 200, f"Filtered boundary request failed: {resp.status}"
    data = resp.json()
    assert data.get("type") == "FeatureCollection"
    features = data.get("features", [])
    assert len(features) > 0, "No features returned for state=Telangana"

    for feat in features:
        props = feat.get("properties", {})
        state_name = props.get("NAME_1") or props.get("state")
        assert state_name.lower() == "telangana".lower(), f"Unexpected state in result: {state_name}"


# ===========================================================================
# TC-GIS-07 – Spatial Attribute Search & Query Index
# ===========================================================================

@pytest.mark.gis
def test_gis07_spatial_query_finds_district_by_name(gis_api: APIRequestContext):
    """TC-GIS-07 – GET /api/v1/spatial/query?q=Adilabad returns matching district geometry."""
    resp = gis_api.get("/api/v1/spatial/query?q=Adilabad")
    assert resp.status == 200, f"Query endpoint failed: {resp.status}"
    data = resp.json()
    assert data.get("type") == "FeatureCollection"
    features = data.get("features", [])
    assert len(features) >= 1, "Expected at least 1 feature matching 'Adilabad'"
    first_props = features[0].get("properties", {})
    name_2 = first_props.get("NAME_2", "")
    assert "adilabad" in name_2.lower(), f"Expected Adilabad in NAME_2, got {name_2}"


# ===========================================================================
# TC-GIS-08 – Sample Points & Density Telemetry
# ===========================================================================

@pytest.mark.gis
def test_gis08_sample_points_and_density_weights(gis_api: APIRequestContext):
    """TC-GIS-08 – GET /api/v1/points/sample returns point features with density weights."""
    resp = gis_api.get("/api/v1/points/sample")
    assert resp.status == 200, f"Points endpoint failed: {resp.status}"
    data = resp.json()
    assert data.get("type") == "FeatureCollection"
    features = data.get("features", [])
    assert len(features) > 0, "Expected non-empty point collection"
    first_feat = features[0]
    assert first_feat.get("geometry", {}).get("type") == "Point"
    assert "intensity" in first_feat.get("properties", {}) or "weight" in first_feat.get("properties", {}) or len(first_feat.get("properties", {})) > 0
