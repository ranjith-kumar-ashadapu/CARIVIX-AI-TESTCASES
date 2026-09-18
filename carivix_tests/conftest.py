"""
conftest.py – CARIVIX-AI Unified Test Suite
============================================

Session-scope fixtures that manage the lifecycle of three FastAPI services
and provide Playwright APIRequestContext clients for each:

  Port 8000 – Backend Data Service   (BACKEND module/api.py)
  Port 8001 – ML Model Inference API (ML module/api.py)
  Port 8002 – ML Items CRUD API      (ML module/fastapi_main.py)

Virtual environment: f:\\CARIVIX\\CARIVIX-AI\\Testing\\.venv
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright, APIRequestContext, Playwright

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
HERE: Path = Path(__file__).resolve().parent          # carivix_tests/
PROJECT_ROOT: Path = HERE.parent                      # Testing/CARIVIX-AI/
TESTING_ROOT: Path = PROJECT_ROOT.parent              # Testing/

VENV_PYTHON: Path = TESTING_ROOT / ".venv" / "Scripts" / "python.exe"

BACKEND_API: Path = PROJECT_ROOT / "BACKEND module" / "api.py"
ML_INFERENCE_API: Path = PROJECT_ROOT / "ML module" / "api.py"
ML_ITEMS_API: Path = PROJECT_ROOT / "ML module" / "fastapi_main.py"

BACKEND_PORT: int = 8000
ML_INFERENCE_PORT: int = 8001
ML_ITEMS_PORT: int = 8002

BACKEND_BASE_URL: str = f"http://127.0.0.1:{BACKEND_PORT}"
ML_INFERENCE_BASE_URL: str = f"http://127.0.0.1:{ML_INFERENCE_PORT}"
ML_ITEMS_BASE_URL: str = f"http://127.0.0.1:{ML_ITEMS_PORT}"


# ---------------------------------------------------------------------------
# Helper – start a uvicorn service in a subprocess
# ---------------------------------------------------------------------------
def _start_service(
    module_path: Path,
    host: str,
    port: int,
    timeout: float = 15.0,
) -> subprocess.Popen:
    """Launch ``uvicorn <module>:app`` and wait until the port is open.

    Parameters
    ----------
    module_path:
        Absolute path to the Python file that contains the FastAPI ``app``.
    host / port:
        Interface and port to bind.
    timeout:
        How long to wait (seconds) before raising ``RuntimeError``.

    Returns
    -------
    subprocess.Popen
        The live subprocess; callers must call ``.terminate()`` + ``.wait()``.
    """
    env = os.environ.copy()
    # Put the service's own directory first so its local imports resolve.
    extra_paths = [str(module_path.parent)]
    # Backend module adds sub-directories to sys.path at runtime; replicate
    # that here via PYTHONPATH so uvicorn picks them up correctly.
    if "BACKEND" in str(module_path):
        backend_root = module_path.parent
        for sub in ("data_acquisition", "data_processing", "database"):
            extra_paths.append(str(backend_root / sub))
    elif "ML" in str(module_path):
        ml_root = module_path.parent
        extra_paths.append(str(ml_root / "src"))

    env["PYTHONPATH"] = os.pathsep.join(extra_paths + [env.get("PYTHONPATH", "")])

    cmd = [
        str(VENV_PYTHON), "-m", "uvicorn",
        f"{module_path.stem}:app",
        f"--host={host}",
        f"--port={port}",
        "--log-level=error",
        f"--app-dir={str(module_path.parent)}",
    ]

    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0

    proc = subprocess.Popen(
        cmd,
        cwd=str(module_path.parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            out, err = proc.communicate()
            raise RuntimeError(
                f"Service {module_path.name} on :{port} exited early.\n"
                f"stdout: {out.decode(errors='ignore')}\n"
                f"stderr: {err.decode(errors='ignore')}"
            )
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return proc          # port is open → service is ready
        except OSError:
            time.sleep(0.25)

    raise RuntimeError(
        f"Service {module_path.name} failed to start on {host}:{port} within {timeout}s"
    )


# ---------------------------------------------------------------------------
# Service fixtures  (session-scope – start once per pytest run)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def backend_service():
    """Start the Backend Data Service on port 8000."""
    proc = _start_service(BACKEND_API, "127.0.0.1", BACKEND_PORT)
    yield
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="session")
def ml_inference_service():
    """Start the ML Model Inference API on port 8001."""
    proc = _start_service(ML_INFERENCE_API, "127.0.0.1", ML_INFERENCE_PORT)
    yield
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="session")
def ml_items_service():
    """Start the ML Items CRUD API (fastapi_main.py) on port 8002."""
    proc = _start_service(ML_ITEMS_API, "127.0.0.1", ML_ITEMS_PORT)
    yield
    proc.terminate()
    proc.wait()


# ---------------------------------------------------------------------------
# Playwright instance  (session-scope)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def playwright_instance() -> Playwright:
    """Return a long-lived Playwright instance for the whole test session."""
    pw = sync_playwright().start()
    yield pw
    pw.stop()


# ---------------------------------------------------------------------------
# Playwright APIRequestContext fixtures – one per service
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def backend_api(playwright_instance: Playwright, backend_service) -> APIRequestContext:
    """Playwright request context pre-configured for the Backend service."""
    ctx = playwright_instance.request.new_context(
        base_url=BACKEND_BASE_URL,
        extra_http_headers={"Accept": "application/json"},
    )
    yield ctx
    ctx.dispose()


@pytest.fixture(scope="session")
def ml_api(playwright_instance: Playwright, ml_inference_service) -> APIRequestContext:
    """Playwright request context pre-configured for the ML Inference service."""
    ctx = playwright_instance.request.new_context(
        base_url=ML_INFERENCE_BASE_URL,
        extra_http_headers={"Accept": "application/json"},
    )
    yield ctx
    ctx.dispose()


@pytest.fixture(scope="session")
def items_api(playwright_instance: Playwright, ml_items_service) -> APIRequestContext:
    """Playwright request context pre-configured for the ML Items CRUD service."""
    ctx = playwright_instance.request.new_context(
        base_url=ML_ITEMS_BASE_URL,
        extra_http_headers={"Accept": "application/json"},
    )
    yield ctx
    ctx.dispose()


# ---------------------------------------------------------------------------
# Utility fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def ml_has_models(ml_api: APIRequestContext) -> bool:
    """Return True if the ML inference service has at least one loaded model."""
    resp = ml_api.get("/health")
    if resp.status == 200:
        return resp.json().get("models_loaded", 0) > 0
    return False


@pytest.fixture
def skip_if_no_models(ml_has_models):
    """Skip the calling test when no ML models are loaded."""
    if not ml_has_models:
        pytest.skip("No ML models loaded – skipping inference test")
