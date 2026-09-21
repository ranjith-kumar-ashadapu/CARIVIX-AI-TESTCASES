"""
run_api_tests.py – CARIVIX-AI Master Test Runner & Automation CLI
===================================================================

Unified push-button automation CLI for backend & spatial integration testing.

Usage:
    python run_api_tests.py                  # Run all core integration tests + report
    python run_api_tests.py --suite backend  # Run Backend Data Service tests
    python run_api_tests.py --suite gis      # Run WebGIS Spatial API tests
    python run_api_tests.py --suite items    # Run Items CRUD API tests
    python run_api_tests.py --clean-ports    # Clean up any lingering service ports
    python run_api_tests.py --report-only    # Re-generate reports from last test run
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Paths
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE
TESTING_ROOT = PROJECT_ROOT.parent
VENV_PYTHON = TESTING_ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)

PORTS = [8000, 8001, 8002, 8003]

SUITE_MAP = {
    "backend": [
        "carivix_tests/backend/test_backend_api.py",
        "carivix_tests/backend/test_backend_pipeline.py",
        "carivix_tests/backend/test_negative_and_gibberish.py",
    ],
    "gis": [
        "carivix_tests/gis/test_gis_api.py",
    ],
    "nlp": [
        "carivix_tests/nlp/test_nlp_api.py",
    ],
    "ml": [
        "carivix_tests/ml/test_ml_feature_pipeline.py",
        "carivix_tests/ml/test_ml_model_api.py",
        "carivix_tests/ml/test_ml_items_api.py",
        "carivix_tests/ml/test_ml_unexpected_inputs.py",
    ],
    "unexpected": [
        "carivix_tests/backend/test_negative_and_gibberish.py",
        "carivix_tests/ml/test_ml_unexpected_inputs.py",
    ],
    "items": [
        "carivix_tests/ml/test_ml_items_api.py",
    ],
    "e2e": [
        "carivix_tests/test_e2e_ai_nlp_gis_workflow.py",
    ],
    "stress": [
        "carivix_tests/stress/test_ml_stress_and_limits.py",
        "carivix_tests/stress/test_nlp_stress_and_limits.py",
        "carivix_tests/stress/test_gis_stress_and_limits.py",
        "carivix_tests/stress/test_backend_stress_and_limits.py",
    ],
    "all": [
        "carivix_tests/backend/test_backend_api.py",
        "carivix_tests/backend/test_backend_pipeline.py",
        "carivix_tests/backend/test_negative_and_gibberish.py",
        "carivix_tests/gis/test_gis_api.py",
        "carivix_tests/nlp/test_nlp_api.py",
        "carivix_tests/ml/test_ml_feature_pipeline.py",
        "carivix_tests/ml/test_ml_model_api.py",
        "carivix_tests/ml/test_ml_items_api.py",
        "carivix_tests/ml/test_ml_unexpected_inputs.py",
        "carivix_tests/test_e2e_ai_nlp_gis_workflow.py",
        "carivix_tests/stress/test_ml_stress_and_limits.py",
        "carivix_tests/stress/test_nlp_stress_and_limits.py",
        "carivix_tests/stress/test_gis_stress_and_limits.py",
        "carivix_tests/stress/test_backend_stress_and_limits.py",
    ],
}


def clean_ports():
    """Kill lingering processes on test ports."""
    print("🧹 [PRE-FLIGHT] Checking for lingering processes on test ports (8000, 8001, 8002, 8003)...")
    for port in PORTS:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                is_open = True
        except OSError:
            is_open = False

        if is_open and sys.platform.startswith("win"):
            print(f"  ⚠️  Port {port} is occupied – releasing...")
            cmd = (
                f'powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort {port} '
                f'-ErrorAction SilentlyContinue | ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"'
            )
            subprocess.run(cmd, shell=True, timeout=5)
            time.sleep(0.3)
    print("✅ [PRE-FLIGHT] Test ports ready.")


def run_tests(suites: list[str], generate_report: bool = True) -> int:
    """Execute pytest on selected suites."""
    clean_ports()

    test_files = []
    for s in suites:
        test_files.extend(SUITE_MAP.get(s, []))

    if not test_files:
        print(f"❌ Error: No test files found for suites: {suites}")
        return 1

    junit_xml = HERE / "carivix_tests" / "reports" / "junit_results.xml"
    junit_xml.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(VENV_PYTHON),
        "-m", "pytest",
        *test_files,
        "-v",
        "--tb=short",
        f"--junitxml={junit_xml}",
    ]

    print("\n" + "=" * 70)
    print(f"🚀 CARIVIX-AI AUTOMATED INTEGRATION TEST RUNNER")
    print(f"📁 Target Suites : {', '.join(suites)}")
    print(f"🐍 Python Venv   : {VENV_PYTHON}")
    print("=" * 70 + "\n")

    start_time = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    total_time = time.perf_counter() - start_time

    print("\n" + "=" * 70)
    print(f"⏱️  Total Execution Time: {total_time:.2f}s")
    print(f"📊 Pytest Exit Code: {res.returncode}")
    print("=" * 70 + "\n")

    if generate_report:
        print("📄 [REPORT] Generating executive test reports...")
        report_script = HERE / "carivix_tests" / "reports" / "generate_test_report.py"
        subprocess.run([str(VENV_PYTHON), str(report_script)], cwd=str(PROJECT_ROOT))

    return res.returncode


def main():
    parser = argparse.ArgumentParser(description="CARIVIX-AI API Integration Test Automation Runner")
    parser.add_argument("--suite", choices=["all", "backend", "gis", "nlp", "ml", "items", "unexpected", "e2e", "stress"], default="all", help="Target test suite")
    parser.add_argument("--clean-ports", action="store_true", help="Clean up ports and exit")
    parser.add_argument("--no-report", action="store_true", help="Skip report generation")
    parser.add_argument("--report-only", action="store_true", help="Only generate report without running tests")

    args = parser.parse_args()

    if args.clean_ports:
        clean_ports()
        return

    if args.report_only:
        report_script = HERE / "carivix_tests" / "reports" / "generate_test_report.py"
        subprocess.run([str(VENV_PYTHON), str(report_script)], cwd=str(PROJECT_ROOT))
        return

    suites = [args.suite]
    code = run_tests(suites, generate_report=not args.no_report)
    sys.exit(code)


if __name__ == "__main__":
    main()
