"""
GIS Team – Test Stubs
======================

Status : PENDING – GIS team delivery
TC-IDs : TC-GIS-01 .. TC-GIS-08  (placeholder – IDs to be confirmed by GIS team)

These tests validate the WebGIS layer, spatial data APIs, and map rendering.
All are marked @pytest.mark.skip until the GIS service/module is delivered.

Run stubs:
    pytest carivix_tests/pending_stubs/test_gis_stub.py -v
"""

import pytest

_REASON = "PENDING – GIS team delivery: spatial service / WebGIS layer not yet available"

# Placeholder – update when GIS service is live
GIS_SERVICE_URL = "http://localhost:8080"


@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_gis01_spatial_api_health_check():
    """
    TC-GIS-01 – GET /gis/health returns HTTP 200 with status: ok.

    TODO: Call the GIS service health endpoint and assert JSON response.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_gis02_district_boundaries_geojson_loads():
    """
    TC-GIS-02 – GET /gis/districts returns valid GeoJSON FeatureCollection.

    Expected:
        HTTP 200.
        Content-Type: application/geo+json.
        'type' == 'FeatureCollection' with ≥ 1 features.

    TODO: Fetch the districts endpoint and validate the GeoJSON structure.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_gis03_choropleth_data_matches_kpi_values():
    """
    TC-GIS-03 – Choropleth colour mapping reflects current KPI values from the backend.

    Expected: Each district's colour weight correlates with the KPI score from /retrieve.

    TODO: Fetch KPI data from backend + choropleth config from GIS API and cross-validate.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_gis04_point_in_polygon_lookup_correct():
    """
    TC-GIS-04 – Given a lat/lon coordinate, the GIS API returns the correct district polygon.

    TODO: POST lat/lon to /gis/lookup and assert the returned district name.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_gis05_map_tile_server_returns_png_tiles():
    """
    TC-GIS-05 – Map tile server responds with PNG tiles for z/x/y requests.

    TODO: GET /tiles/{z}/{x}/{y}.png and assert Content-Type: image/png.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_gis06_spatial_filter_by_bounding_box():
    """
    TC-GIS-06 – Query districts within a bounding box returns only overlapping features.

    TODO: POST bbox coords to /gis/districts?bbox=... and assert returned features.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_gis07_wms_getmap_request_returns_image():
    """
    TC-GIS-07 – WMS GetMap request returns a valid image response.

    TODO: Issue a WMS GetMap request and assert HTTP 200 + image MIME type.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_gis08_geojson_coordinate_reference_system_wgs84():
    """
    TC-GIS-08 – GeoJSON from the GIS API uses WGS84 (EPSG:4326) coordinate system.

    TODO: Inspect the 'crs' field of the GeoJSON response.
    """
    raise NotImplementedError
