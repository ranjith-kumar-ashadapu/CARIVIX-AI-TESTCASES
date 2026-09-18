from __future__ import annotations
import sys
sys.path.insert(0, "data_acquisition")
sys.path.insert(0, "data_processing")
sys.path.insert(0, "database")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any

import pandas as pd
from processing_workflow import ProcessingWorkflow, ProcessingWorkflowError
from ingestion_service import DataIngestionService, IngestionValidationError
from database_service import DatabaseService, DatabaseServiceError
from data_validation import DataValidator, Schema, ColumnRule

import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
api_logger = logging.getLogger("carivix.api")

app = FastAPI(title="CARIVIX AI Data Service API")


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    api_logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms)"
    )
    return response


class ProcessRequest(BaseModel):
    records: list[dict[str, Any]]
    profile: str


class ProcessResponse(BaseModel):
    row_count: int
    columns: list[str]
    data: list[dict[str, Any]]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sources")
def list_sources():
    svc = DataIngestionService()
    return svc.list_sources()


@app.post("/process", response_model=ProcessResponse)
def process_data(request: ProcessRequest):
    try:
        df = pd.DataFrame(request.records)
        wf = ProcessingWorkflow()
        result = wf.run_with_profile(df, request.profile)
        return ProcessResponse(
            row_count=len(result),
            columns=list(result.columns),
            data=result.to_dict(orient="records"),
        )
    except ProcessingWorkflowError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    

class StoreRequest(BaseModel):
    table: str
    records: list[dict[str, Any]]


class StoreResponse(BaseModel):
    rows_written: int


@app.post("/store", response_model=StoreResponse)
def store_data(request: StoreRequest):
    if not request.records:
        raise HTTPException(status_code=400, detail="Cannot store an empty records list")
    try:
        with DatabaseService("carivix_api.db") as db:
            sample = request.records[0]
            columns = {
                key: ("REAL" if isinstance(val, float) else "INTEGER" if isinstance(val, int) else "TEXT")
                for key, val in sample.items()
            }
            columns["id"] = "INTEGER PRIMARY KEY AUTOINCREMENT"
            db.db.create_table(request.table, columns, if_not_exists=True)
            count = db.write(request.table, request.records)
            return StoreResponse(rows_written=count)
    except DatabaseServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    

class RetrieveResponse(BaseModel):
    row_count: int
    data: list[dict[str, Any]]


@app.get("/retrieve/{table}", response_model=RetrieveResponse)
def retrieve_data(
    table: str,
    filter_field: Optional[str] = None,
    filter_value: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
):
    try:
        with DatabaseService("carivix_api.db") as db:
            where = {filter_field: filter_value} if filter_field and filter_value else None
            all_rows = db.read(table, where=where)
            paginated = all_rows[offset:offset + limit] if limit else all_rows[offset:]
            return RetrieveResponse(row_count=len(paginated), data=paginated)
    except DatabaseServiceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    

class PipelineRequest(BaseModel):
    records: list[dict[str, Any]]
    profile: str
    table: str


class PipelineResponse(BaseModel):
    row_count: int
    columns: list[str]
    rows_written: int
    table: str


@app.post("/pipeline", response_model=PipelineResponse)
def run_pipeline(request: PipelineRequest):
    try:
        df = pd.DataFrame(request.records)
        wf = ProcessingWorkflow()
        processed = wf.run_with_profile(df, request.profile)
    except ProcessingWorkflowError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        with DatabaseService("carivix_api.db") as db:
            sample = processed.to_dict(orient="records")[0]
            columns = {
                key: ("REAL" if isinstance(val, float) else "INTEGER" if isinstance(val, int) else "TEXT")
                for key, val in sample.items()
            }
            columns["id"] = "INTEGER PRIMARY KEY AUTOINCREMENT"
            db.db.create_table(request.table, columns, if_not_exists=True)
            written = db.write(request.table, processed.to_dict(orient="records"))
        return PipelineResponse(
            row_count=len(processed),
            columns=list(processed.columns),
            rows_written=written,
            table=request.table,
        )
    except DatabaseServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    

DTYPE_MAP = {"str": str, "float": float, "int": int}


class ValidationRule(BaseModel):
    name: str
    dtype: Optional[str] = None
    required: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class ValidateRequest(BaseModel):
    records: list[dict[str, Any]]
    rules: list[ValidationRule]


class ValidateResponse(BaseModel):
    total_rows: int
    valid_rows: int
    duplicate_row_count: int
    missing_value_counts: dict[str, int]
    row_error_count: int
    is_valid: bool


@app.post("/validate", response_model=ValidateResponse)
def validate_data(request: ValidateRequest):
    if not request.records:
        raise HTTPException(status_code=400, detail="Cannot validate an empty records list")
    try:
        column_rules = [
            ColumnRule(
                name=r.name,
                dtype=DTYPE_MAP.get(r.dtype) if r.dtype else None,
                required=r.required,
                min_value=r.min_value,
                max_value=r.max_value,
            )
            for r in request.rules
        ]
        schema = Schema(columns=column_rules)
        validator = DataValidator(schema)
        df = pd.DataFrame(request.records)
        report = validator.validate(df)
        return ValidateResponse(
            total_rows=report.total_rows,
            valid_rows=report.valid_rows,
            duplicate_row_count=report.duplicate_row_count,
            missing_value_counts=report.missing_value_counts,
            row_error_count=len(report.row_errors),
            is_valid=report.is_valid,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")