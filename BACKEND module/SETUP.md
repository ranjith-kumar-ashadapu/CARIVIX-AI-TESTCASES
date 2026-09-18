# CARIVIX AI — Local Environment Setup

This is the consolidated Python/R developer codebase, covering the full data-service layer and API built across Sprints 1 through 4. Every module here has been tested and confirmed working, with 70 automated tests covering the pipeline and the API.

## Folder structure

```
carivix_ai/
├── api.py                        FastAPI service layer, 7 endpoints
├── config/
│   ├── data_sources.yaml         master config for all data sources
│   └── processing_profiles.yaml  named processing profiles
├── data_acquisition/
│   ├── importers/                CSV, Excel, JSON, API importers
│   ├── utils/                    config loader, logger
│   ├── ingestion_service.py      unified ingestion entry point
│   └── main.py                   CLI entry point
├── data_processing/
│   ├── data_validation.py        schema-driven validator
│   ├── error_handling.py         custom exceptions, safe_run retry decorator
│   ├── logging_config.py         centralized logger factory
│   ├── data_transformation.py    cleaning, transformation, storage
│   ├── preprocessing_optimized.py performance-optimized versions
│   └── processing_workflow.py    connects cleaning, transformation, and optimization as one workflow
├── database/
│   ├── db_crud.py                SQLite CRUD manager, swappable to Postgres or MySQL
│   └── database_service.py       connectivity, read/write, logging, error handling
├── nlp/                          NLP helper functions
├── predictive_analytics/         forecasting helpers (ARIMA, Prophet, growth rate)
├── visualization/                reusable plotting functions
├── tests/                        70 automated tests across all modules and the API
└── docs/                         architecture, data flow, and API reference docs
```

## Setup steps (run these in order)

1. Install Python 3.10 or later.

2. Create and activate a virtual environment in the project root:
   python3 -m venv venv
   Mac/Linux: source venv/bin/activate
   Windows: venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Verify the backend works by running the full test suite:
   python3 -m pytest tests/ -v
   This should end with all 70 tests passing.

5. Start the API server:
   uvicorn api:app --reload
   Then confirm it's running by visiting http://127.0.0.1:8000/docs in a browser, which shows all 7 endpoints with interactive documentation.
   
   ## What's built and tested

Four connected services: Data Ingestion, Database, Processing Workflow, and a full API layer wrapping all of them. Automated test coverage spans unit tests, integration tests across the full pipeline, a concurrency stress test, and real-dataset performance tests at 500 and 5,000 rows.

## Known gaps going into Sprint 5

There is currently no authentication on any API endpoint. The database is a single fixed SQLite file rather than a configurable connection. Predictive analytics functions exist but are not yet wrapped into a service or exposed through the API. GIS and AI/ML integration are planned for this sprint and not yet connected.