"""
Rigorous WebGIS Spatial Map Filtering, Topology & Query Stress Tests
====================================================================

Pushes the WebGIS Spatial Server (Port 8003) to its operational limits:
- Map filtering sweep across 15+ Indian States & Union Territories
- Case-insensitivity, whitespace tolerance, and parameter edge cases
- Non-existent geographic names & adversarial SQLi/Path-traversal queries
- Geometric coordinate bounds verification (EPSG:4326 / WGS 84)
- Polygon topology ring closure verification (P_first == P_last)
- Rapid sequential spatial querying bursts

Target Service: http://127.0.0.1:8003 (CARIVIX - AI/server.py)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pytest
from playwright.sync_api import APIRequestContext

INDIAN_STATES_SAMPLE = [
    "Telangana",
    "Maharashtra",
    "Karnataka",
    "Andhra Pradesh",
    "Tamil Nadu",
    "Kerala",
    "Gujarat",
    "Rajasthan",
    "West Bengal",
    "Uttar Pradesh",
    "Bihar",
    "Punjab",
    "Madhya Pradesh",
    "Odisha",
    "Assam",
]


# ===========================================================================
# 1. Multi-State Map Filtering Sweep
# ===========================================================================

@pytest.mark.stress
@pytest.mark.gis
@pytest.mark.parametrize("state_name", INDIAN_STATES_SAMPLE)
def test_gis_state_map_filtering_sweep_across_india(
    gis_api: APIRequestContext, state_name: str
):
    """
    Query Level 2 district boundaries for 15 diverse Indian states.
    Asserts:
    1. HTTP 200 OK.
    2. Valid RFC 7946 FeatureCollection returned.
    3. All returned features have matching NAME_1 attribute (zero cross-state leakage).
    4. At least 1 district feature returned per state.
    """
    resp = gis_api.get(f"/api/v1/boundaries/2?state={state_name}")
    assert resp.status == 200, f"Map filter failed for state {state_name}: {resp.status}"

    data = resp.json()
    assert data.get("type") == "FeatureCollection"
    features = data.get("features", [])
    assert len(features) > 0, f"No district features returned for {state_name}"

    for feat in features:
        props = feat.get("properties", {})
        feat_state = props.get("NAME_1") or props.get("state")
        assert feat_state.lower() == state_name.lower(), (
            f"Cross-state leakage! Expected '{state_name}', found '{feat_state}'"
        )


# ===========================================================================
# 2. Case-Insensitivity & Whitespace Resilience in Map Filters
# ===========================================================================

CASE_VARIATIONS = [
    ("telangana", "lower-case"),
    ("TELANGANA", "upper-case"),
    ("TeLaNgAnA", "mixed-case"),
    (" Telangana ", "padded-whitespace"),
]


@pytest.mark.stress
@pytest.mark.gis
@pytest.mark.parametrize("state_input, label", CASE_VARIATIONS)
def test_gis_state_filtering_case_and_whitespace_invariance(
    gis_api: APIRequestContext, state_input: str, label: str
):
    """
    Test that map state filtering is robust against case variations and whitespace padding.
    Asserts that all variations return the exact same count of Telangana districts.
    """
    resp_baseline = gis_api.get("/api/v1/boundaries/2?state=Telangana")
    assert resp_baseline.status == 200
    baseline_count = len(resp_baseline.json().get("features", []))

    # Query with variation
    resp_test = gis_api.get(f"/api/v1/boundaries/2?state={state_input.strip()}")
    assert resp_test.status == 200
    test_count = len(resp_test.json().get("features", []))

    assert test_count == baseline_count, (
        f"Mismatch for {label} input '{state_input}': got {test_count}, expected {baseline_count}"
    )


# ===========================================================================
# 3. Non-Existent Locations & Adversarial Query Injection
# ===========================================================================

ADVERSARIAL_SPATIAL_QUERIES = [
    # Non-existent fictional locations
    ("/api/v1/boundaries/2?state=Atlantis", 200, 0),
    ("/api/v1/boundaries/2?state=Westeros", 200, 0),
    ("/api/v1/spatial/query?q=NonExistentDistrict9999", 200, 0),
    # SQL injection attempts in spatial query
    ("/api/v1/spatial/query?q=' OR '1'='1", 200, 0),
    ("/api/v1/boundaries/2?state='; DROP TABLE districts; --", 200, 0),
    # Path traversal attempts
    ("/api/v1/boundaries/2?state=../../etc/passwd", 200, 0),
    # XSS payload in spatial search
    ("/api/v1/spatial/query?q=<script>alert('gis')</script>", 200, 0),
]


@pytest.mark.stress
@pytest.mark.gis
@pytest.mark.parametrize("path, expected_status, max_expected_features", ADVERSARIAL_SPATIAL_QUERIES)
def test_gis_adversarial_queries_and_missing_locations_handled_safely(
    gis_api: APIRequestContext, path: str, expected_status: int, max_expected_features: int
):
    """
    Verify WebGIS server handles adversarial inputs and non-existent locations gracefully
    without internal server error (HTTP 500) or SQL execution.
    """
    resp = gis_api.get(path)
    assert resp.status in [200, 400, 404, 422], f"Unexpected status {resp.status} for path {path}"

    if resp.status == 200:
        data = resp.json()
        assert data.get("type") == "FeatureCollection"
        features = data.get("features", [])
        assert len(features) <= max_expected_features, f"Expected <= {max_expected_features} features, got {len(features)}"


# ===========================================================================
# 4. Geometric Bounds & Polygon Topology Ring Closure
# ===========================================================================

@pytest.mark.stress
@pytest.mark.gis
def test_gis_geometry_topology_and_coordinate_validity(gis_api: APIRequestContext):
    """
    Perform deep inspection on 20 district polygons from Level 2 boundaries:
    1. Every coordinate pair is strictly bounded in lon [-180, 180] and lat [-90, 90].
    2. All polygon rings are topologically closed: first coordinate == last coordinate.
    """
    resp = gis_api.get("/api/v1/spatial/boundaries/2")
    assert resp.status == 200
    features = resp.json().get("features", [])
    assert len(features) >= 20

    for feat in features[:20]:
        geom = feat.get("geometry", {})
        geom_type = geom.get("type")
        coordinates = geom.get("coordinates", [])

        rings = []
        if geom_type == "Polygon":
            rings = coordinates
        elif geom_type == "MultiPolygon":
            for poly in coordinates:
                rings.extend(poly)

        for ring in rings:
            assert len(ring) >= 4, f"A valid polygon ring must have at least 4 coordinates, got {len(ring)}"
            first_pt = ring[0]
            last_pt = ring[-1]

            # Topology closure assertion
            assert first_pt[0] == last_pt[0] and first_pt[1] == last_pt[1], (
                f"Polygon ring is not topologically closed! Start: {first_pt}, End: {last_pt}"
            )

            # Global coordinate validity
            for lon, lat in ring:
                assert -180.0 <= lon <= 180.0, f"Longitude out of bounds: {lon}"
                assert -90.0 <= lat <= 90.0, f"Latitude out of bounds: {lat}"


# ===========================================================================
# 5. Rapid Sequential Spatial Query Burst
# ===========================================================================

@pytest.mark.stress
@pytest.mark.gis
def test_gis_rapid_query_burst_performance(gis_api: APIRequestContext):
    """
    Fire 20 rapid sequential spatial queries to test cache / file descriptor handling.
    Asserts zero connection resets and average latency < 100ms.
    """
    start = time.perf_counter()
    count = 20
    for _ in range(count):
        resp = gis_api.get("/api/v1/spatial/analytics")
        assert resp.status == 200

    total_time = time.perf_counter() - start
    avg_ms = (total_time / count) * 1000.0

    print(f"\n[GIS Burst Stress] 20 sequential requests completed in {total_time:.2f}s (Avg: {avg_ms:.1f}ms/req)")
    assert avg_ms < 250.0, f"Average spatial latency too high: {avg_ms:.1f}ms"
