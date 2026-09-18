"""
CARIVIX AI - Week 4 Data Service Layer Demo
Demonstrates the full pipeline: ingestion validation -> processing -> database storage -> retrieval
Run with: python3 week3_demo.py
"""
import sys
sys.path.insert(0, "data_acquisition")
sys.path.insert(0, "data_processing")
sys.path.insert(0, "database")

import pandas as pd
from ingestion_service import DataIngestionService, IngestionValidationError
from processing_workflow import ProcessingWorkflow
from database_service import DatabaseService

print("=" * 60)
print("STEP 1: Data Ingestion Service - source discovery")
print("=" * 60)
ingest_svc = DataIngestionService()
print(ingest_svc.list_sources())

print("\n" + "=" * 60)
print("STEP 2: Input validation - rejecting a bad request")
print("=" * 60)
try:
    ingest_svc.ingest("bad_category", "anything")
except IngestionValidationError as e:
    print(f"Validation correctly rejected bad input: {e}")

print("\n" + "=" * 60)
print("STEP 3: Processing Workflow - profile-driven pipeline run")
print("=" * 60)
messy_sample = pd.DataFrame({
    "Company Name": ["Acme", "Acme", "Beta", None],
    "Revenue": [100.0, 100.0, "not_a_number", 250.0],
    "Region": ["South", "South", "north", "North"],
    "Report Date": ["2026-01-05", "2026-01-05", "2026-02-10", "2026-03-01"],
})
wf = ProcessingWorkflow()
processed = wf.run_with_profile(messy_sample, "company_financials")
print(processed)

print("\n" + "=" * 60)
print("STEP 4: Database Service - store and retrieve processed data")
print("=" * 60)
with DatabaseService(":memory:") as db:
    columns = {col: "REAL" if processed[col].dtype.kind == "f" else "TEXT" for col in processed.columns}
    db.db.create_table("demo_companies", columns)
    db.write("demo_companies", processed.to_dict(orient="records"))
    rows = db.read("demo_companies")
    print(f"Retrieved {len(rows)} row(s) from database:")
    for row in rows:
        print(" ", row)

print("\n" + "=" * 60)
print("DEMO COMPLETE - all four services connected and working")
print("=" * 60)