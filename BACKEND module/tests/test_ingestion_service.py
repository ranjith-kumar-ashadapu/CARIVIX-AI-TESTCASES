import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_acquisition"))
from ingestion_service import DataIngestionService, IngestionValidationError


def test_list_sources_returns_all_categories():
    svc = DataIngestionService()
    sources = svc.list_sources()
    assert "csv_sources" in sources
    assert "excel_sources" in sources
    assert "json_sources" in sources
    assert "api_sources" in sources


def test_list_sources_filters_by_category():
    svc = DataIngestionService()
    sources = svc.list_sources(category="csv_sources")
    assert list(sources.keys()) == ["csv_sources"]
    assert "census_data" in sources["csv_sources"]


def test_ingest_routes_to_csv_importer_and_fails_cleanly_without_file():
    svc = DataIngestionService()
    with pytest.raises(Exception):
        svc.ingest("csv_sources", "census_data")


def test_ingest_routes_to_excel_importer_and_fails_cleanly_without_file():
    svc = DataIngestionService()
    with pytest.raises(Exception):
        svc.ingest("excel_sources", "company_financials")


def test_ingest_routes_to_json_importer_and_fails_cleanly_without_file():
    svc = DataIngestionService()
    with pytest.raises(Exception):
        svc.ingest("json_sources", "government_open_data")


def test_invalid_category_raises_validation_error():
    svc = DataIngestionService()
    with pytest.raises(IngestionValidationError):
        svc.ingest("bad_category", "census_data")


def test_unknown_source_name_raises_validation_error():
    svc = DataIngestionService()
    with pytest.raises(IngestionValidationError):
        svc.ingest("csv_sources", "not_a_real_source")


def test_empty_category_raises_validation_error():
    svc = DataIngestionService()
    with pytest.raises(IngestionValidationError):
        svc.ingest("", "census_data")