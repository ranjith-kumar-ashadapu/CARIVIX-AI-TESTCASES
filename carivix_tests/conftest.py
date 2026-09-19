"""
conftest.py – CARIVIX-AI Unified Test Suite
============================================

Session-scope fixtures that manage the lifecycle of four FastAPI services
and provide Playwright APIRequestContext clients for each:

  Port 8000 – Backend Data Service   (BACKEND module/api.py)
  Port 8001 – ML Model Inference API (ML module/api.py)
  Port 8002 – ML Items CRUD API      (ML module/fastapi_main.py)
  Port 8003 – WebGIS Spatial Service (CARIVIX - AI/server.py)

Virtual environment: f:\\CARIVIX\\CARIVIX-AI\\Testing\\.venv
"""

from __future__ import annotations

import io
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Tuple

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
GIS_API: Path = TESTING_ROOT.parent / "CARIVIX - AI" / "server.py"

BACKEND_PORT: int = 8000
ML_INFERENCE_PORT: int = 8001
ML_ITEMS_PORT: int = 8002

BACKEND_BASE_URL: str = f"http://127.0.0.1:{BACKEND_PORT}"
ML_INFERENCE_BASE_URL: str = f"http://127.0.0.1:{ML_INFERENCE_PORT}"
ML_ITEMS_BASE_URL: str = f"http://127.0.0.1:{ML_ITEMS_PORT}"


# ---------------------------------------------------------------------------
# Port management & Helper – start a uvicorn service in a subprocess
# ---------------------------------------------------------------------------
def _clean_port(port: int) -> None:
    """Ensure port is not occupied by a stale process."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            pass
    except OSError:
        return  # Port is free
    if sys.platform.startswith("win"):
        try:
            cmd = (
                f'powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort {port} '
                f'-ErrorAction SilentlyContinue | ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"'
            )
            subprocess.run(cmd, shell=True, timeout=5)
            time.sleep(0.5)
        except Exception:
            pass


def _start_service(
    module_path: Path,
    host: str,
    port: int,
    timeout: float = 60.0,
) -> Tuple[subprocess.Popen, Any]:
    """Launch ``uvicorn <module>:app``, stream logs to file (preventing pipe deadlocks),
    and wait until the port is open."""
    _clean_port(port)

    env = os.environ.copy()
    extra_paths = [str(module_path.parent)]
    if "BACKEND" in str(module_path):
        backend_root = module_path.parent
        for sub in ("data_acquisition", "data_processing", "database"):
            extra_paths.append(str(backend_root / sub))
    elif "ML" in str(module_path):
        ml_root = module_path.parent
        extra_paths.append(str(ml_root / "src"))
    elif "CARIVIX - AI" in str(module_path):
        extra_paths.append(str(module_path.parent))

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

    logs_dir = HERE / "reports" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_dir / f"{module_path.stem}_{port}.log"
    log_file = open(log_file_path, "w", encoding="utf-8", buffering=1)

    proc = subprocess.Popen(
        cmd,
        cwd=str(module_path.parent),
        env=env,
        stdout=log_file,
        stderr=log_file,
        creationflags=creationflags,
    )

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            log_file.flush()
            with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
                log_content = f.read()
            log_file.close()
            raise RuntimeError(
                f"Service {module_path.name} on :{port} exited early.\n"
                f"Log:\n{log_content}"
            )
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return proc, log_file
        except OSError:
            time.sleep(0.25)

    log_file.flush()
    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
        log_content = f.read()
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
    log_file.close()
    raise RuntimeError(
        f"Service {module_path.name} failed to start on {host}:{port} within {timeout}s.\n"
        f"Log:\n{log_content}"
    )


# ---------------------------------------------------------------------------
# Service fixtures  (session-scope – start once per pytest run)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def backend_service():
    """Start the Backend Data Service on port 8000."""
    proc, log_file = _start_service(BACKEND_API, "127.0.0.1", BACKEND_PORT)
    yield
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    log_file.close()


@pytest.fixture(scope="session")
def ml_inference_service():
    """Start the ML Model Inference API on port 8001."""
    proc, log_file = _start_service(ML_INFERENCE_API, "127.0.0.1", ML_INFERENCE_PORT)
    yield
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    log_file.close()


@pytest.fixture(scope="session")
def ml_items_service():
    """Start the ML Items CRUD API (fastapi_main.py) on port 8002."""
    proc, log_file = _start_service(ML_ITEMS_API, "127.0.0.1", ML_ITEMS_PORT)
    yield
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    log_file.close()


@pytest.fixture(scope="session")
def gis_service():
    """GIS service stub – GIS is an external module not located in this repository."""
    yield None


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


@pytest.fixture(scope="session")
def gis_api():
    """GIS API context stub – GIS is an external module not located in this repository."""
    pytest.skip("GIS spatial module is an external service not present in this repository.")


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
