import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_processing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "database"))
import pandas as pd
from processing_workflow import ProcessingWorkflow
from database_service import DatabaseService


def test_processing_to_database_end_to_end():
    sample = pd.DataFrame({
        "Company Name": ["Acme", "Acme", "Beta"],
        "Revenue": [100.0, 100.0, 250.0],
        "Region": ["South", "South", "north"],
    })

    wf = ProcessingWorkflow()
    processed = wf.run(sample, normalize_cols=["revenue"], categorical_cols=["region"])

    with DatabaseService(":memory:") as db:
        columns = {col: "REAL" if processed[col].dtype.kind == "f" else "TEXT" for col in processed.columns}
        db.db.create_table("processed_companies", columns)
        db.write("processed_companies", processed.to_dict(orient="records"))
        rows = db.read("processed_companies")

    assert len(rows) == 2
    assert rows[0]["company_name"] in ("Acme", "Beta")


def test_ingestion_to_processing_handles_missing_source_gracefully():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_acquisition"))
    from ingestion_service import DataIngestionService, IngestionValidationError
    import pytest

    svc = DataIngestionService()
    with pytest.raises(Exception):
        svc.ingest("csv_sources", "census_data")