# carivix_tests – CARIVIX-AI Unified Playwright Test Suite

One folder. All tests. Playwright-Python (sync API).

---

## Folder Structure

```
carivix_tests/
├── conftest.py               ← Session fixtures: 3 services + Playwright contexts
├── backend/
│   ├── test_backend_api.py   ← TC-BE-01..06 + extended scenarios
│   └── test_backend_pipeline.py ← Full pipeline + concurrency
├── ml/
│   ├── test_ml_model_api.py  ← TC-ML-01,02,06,07 + extended (loan prediction)
│   ├── test_ml_items_api.py  ← ML Items CRUD (fastapi_main.py)
│   └── test_ml_rag_pipeline.py ← TC-ML-03,04,05 (RAG unit tests)
└── pending_stubs/
    ├── test_nlp_stub.py       ← TC-NLP-01..08 (SKIPPED – NLP team pending)
    ├── test_fullstack_stub.py ← TC-FS-01..06  (SKIPPED – Full-Stack pending)
    ├── test_gis_stub.py       ← GIS placeholders (SKIPPED)
    └── test_app_stub.py       ← App placeholders  (SKIPPED)
```

---

## Services Under Test

| Service | Entry point | Port | Status |
|---|---|---|---|
| Backend Data Service | `BACKEND module/api.py` | 8000 | ✅ Active |
| ML Model Inference | `ML module/api.py` | 8001 | ✅ Active |
| ML Items CRUD | `ML module/fastapi_main.py` | 8002 | ✅ Active |
| NLP Service | pending | — | ⏳ Stub |
| Full-Stack / UI | pending | — | ⏳ Stub |
| GIS Service | pending | — | ⏳ Stub |
| App (mobile/desktop) | pending | — | ⏳ Stub |

---

## Prerequisites

```powershell
# Activate the virtual environment
& "f:\CARIVIX\CARIVIX-AI\Testing\.venv\Scripts\Activate.ps1"

# Install / verify dependencies
python -m pip install playwright pytest pytest-playwright
playwright install chromium
```

---

## Running the Tests

```powershell
# Navigate to the project root
cd "f:\CARIVIX\CARIVIX-AI\Testing\CARIVIX-AI"

# Run all active tests (stubs auto-skip)
pytest

# Run only Backend tests
pytest carivix_tests/backend/ -v

# Run only ML inference tests
pytest carivix_tests/ml/test_ml_model_api.py -v

# Run only ML Items CRUD tests
pytest carivix_tests/ml/test_ml_items_api.py -v

# Run RAG pipeline tests
pytest carivix_tests/ml/test_ml_rag_pipeline.py -v

# Run everything including stubs (to see skip summary)
pytest carivix_tests/ -v

# Run by marker
pytest -m backend -v
pytest -m "ml and smoke" -v
pytest -m pending -v    # shows all pending stubs

# Show only failures
pytest --tb=long -q
```

---

## TC-ID Coverage

| TC-ID | Test file | Status |
|---|---|---|
| TC-BE-01 | `backend/test_backend_api.py` | Active |
| TC-BE-02 | `backend/test_backend_api.py` | Active |
| TC-BE-03 | `backend/test_backend_api.py` + `test_backend_pipeline.py` | Active |
| TC-BE-04 | `backend/test_backend_api.py` | Active |
| TC-BE-05 | `backend/test_backend_pipeline.py` | Active |
| TC-BE-06 | `backend/test_backend_api.py` | Active |
| TC-ML-01 | `ml/test_ml_model_api.py` | Active |
| TC-ML-02 | `ml/test_ml_model_api.py` | Active |
| TC-ML-03 | `ml/test_ml_rag_pipeline.py` | Active |
| TC-ML-04 | `ml/test_ml_rag_pipeline.py` | Active |
| TC-ML-05 | `ml/test_ml_rag_pipeline.py` | Active |
| TC-ML-06 | `ml/test_ml_model_api.py` | Active |
| TC-ML-07 | `ml/test_ml_model_api.py` (smoke) | Active |
| TC-NLP-01..08 | `pending_stubs/test_nlp_stub.py` | Skipped |
| TC-FS-01..06 | `pending_stubs/test_fullstack_stub.py` | Skipped |
| TC-GIS-* | `pending_stubs/test_gis_stub.py` | Skipped |
| TC-APP-* | `pending_stubs/test_app_stub.py` | Skipped |

---

## Adding Tests for Pending Teams

1. Open the relevant stub file in `pending_stubs/`.
2. Remove the `@pytest.mark.skip` decorator.
3. Implement the test body.
4. Move the function to the appropriate active test file (or create a new one).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Port already in use` | `taskkill /F /IM uvicorn.exe` |
| `playwright not found` | `pip install pytest-playwright && playwright install` |
| `models_loaded: 0` on ML health | ML models in `ML module/models/` – check `.pkl` files exist |
| Import errors | Verify `.venv` is activated before running pytest |
