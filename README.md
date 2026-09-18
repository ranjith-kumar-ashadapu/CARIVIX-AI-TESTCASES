# CARIVIX‑AI End‑to‑End Test Suite

This repository contains a **Playwright‑Python** end‑to‑end (E2E) test suite
that validates the two FastAPI services shipped with CARIVIX‑AI:

* **Backend service** – `CARIVIX-AI/BACKEND module/carivix-ai-backend-main/api.py`
* **ML service**      – `CARIVIX-AI/ML module/fastapi_main.py`

The test cases are defined in an Excel matrix (`TEST CASES - CARIVIX‑AI.xlsx`).
A small helper script converts that matrix into a JSON file that the Playwright
tests consume.

---

## Prerequisites

1. **Python 3.14** (the project uses a virtual environment located at
   `F:\CARIVIX\CARIVIX‑AI\Testing\env`).
2. **Playwright browsers** – after installing the Python dependencies run
   `playwright install` (this has already been executed for you).

## Setup

```powershell
# Activate the virtual environment
& "F:\CARIVIX\CARIVIX-AI\Testing\env\Scripts\Activate.ps1"

# Upgrade pip (optional but recommended)
python -m pip install --upgrade pip

# Install the required packages (already done during the previous step)
python -m pip install openpyxl playwright pytest

# Install Playwright browsers (already done)
playwright install
```

## Generate the test data

The Excel matrix lives at
`F:\CARIVIX\CARIVIX‑AI\Testing\CARIVIX‑AI\TEST CASES - CARIVIX‑AI.xlsx`.
Run the helper script to create the JSON file consumed by the tests:

```powershell
python scripts\generate_tests.py
```

You should see output similar to:

```
Wrote 42 test cases to CARIVIX-AI\tests\e2e\test_cases.json
```

---

## Running the tests

The test suite launches both FastAPI services automatically via pytest fixtures
defined in `tests/e2e/conftest.py`.  Execute the suite with:

```powershell
pytest -s tests/e2e
```

* `-s` ensures that any `print` output from the fixture (e.g., service start
  messages) is shown in the console.
* All tests are written with Playwright’s sync API.  They will run headless by
  default; you can set the `HEADLESS` environment variable to `0` if you want to
  watch the browser:

```powershell
$env:HEADLESS = 0
pytest -s tests/e2e
```

---

## Test report

After the run, pytest will display a summary similar to:

```
============================= test session starts ==============================
platform win32 -- Python 3.14.7, pytest-8.2.2, pluggy-1.5.0
rootdir: F:\CARIVIX\CARIVIX-AI\Testing\CARIVIX-AI
collected 42 items

tests/e2e/test_api.py ..F..F.............................................. [100%]

========================== 2 failed, 40 passed in 12.34s ======================
```

* **Failed tests** indicate mismatches between the actual service response and the
  expectations defined in the Excel matrix.  Review the failure output to locate
  the offending endpoint and payload.
* **Passed tests** confirm that the service behaved exactly as specified.

---

## Adding new test cases

1. Append a new row to the Excel file following the column definitions.
2. Re‑run `python scripts\generate_tests.py` to refresh the JSON data.
3. Execute the pytest command again – the new case will be automatically
   included.

---

### Troubleshooting

* **Port already in use** – Ensure no other process is bound to `8000` or `8001`
  before running the suite.  You can stop stray `uvicorn` instances with
  `taskkill /F /IM uvicorn.exe` on Windows.
* **Missing browsers** – Re‑run `playwright install`.
* **Import errors** – Verify that the virtual environment is activated and that
  `openpyxl`, `playwright` and `pytest` are installed.

---

Happy testing!
