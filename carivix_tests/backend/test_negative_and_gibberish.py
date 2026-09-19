"""
Negative, Gibberish & Intentional Failure Tests
=================================================

Three categories of tests in this file:

  SECTION A – Gibberish / Malformed inputs
      API must safely REJECT all garbage – these tests PASS when the API
      responds with 4xx (the correct behaviour).

  SECTION B – SQL-injection, Unicode & Security inputs
      API must not crash or leak data – tests PASS when service stays stable.

  SECTION C – Intentional Failures  (@pytest.mark.xfail)
      Tests that are deliberately asserting WRONG expectations.
      These are marked xfail so pytest records them as "Expected Failures"
      (XFAIL) without failing the whole suite.
      Purpose: demonstrate the failure-detection capability of the suite.

  SECTION D – Genuine Failures (no xfail decorator)
      A small number of tests with intentionally wrong assertions LEFT
      without xfail so they appear as FAIL in the evidence report.
      Purpose: prove the CI pipeline catches real regressions.

Markers used:
  @pytest.mark.neg      – negative / rejection tests
  @pytest.mark.xfail    – intentional expected failures

Run:
    pytest carivix_tests/backend/test_negative_and_gibberish.py -v
    pytest carivix_tests/backend/test_negative_and_gibberish.py -m neg -v
"""

from __future__ import annotations

import json
import string
import random
import pytest
from playwright.sync_api import APIRequestContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _random_string(length: int = 50) -> str:
    """Generate a random printable-ASCII string."""
    return "".join(random.choices(string.printable, k=length))


VALID_RECORDS = [
    {"Company Name": "Alpha Corp", "Revenue": 100_000.0, "Region": "South", "Report Date": "2026-01-01"},
]
VALID_PROFILE = "company_financials"


# ===========================================================================
# SECTION A – Gibberish / Malformed input tests  (all should PASS)
# ===========================================================================

@pytest.mark.neg
@pytest.mark.backend
def test_neg_a01_gibberish_profile_name_returns_400(backend_api: APIRequestContext):
    """
    TC-NEG-A01 – Formatted records + completely gibberish profile name.

    Input  : Valid JSON records, profile = '!!!@@@###$$$%%% INVALID_PROFILE ^^^&&&***'
    Expect : HTTP 400 – server rejects unknown profile, does not crash.
    """
    response = backend_api.post(
        "/process",
        data={
            "records": VALID_RECORDS,
            "profile": "!!!@@@###$$$%%% INVALID_PROFILE ^^^&&&***",
        },
        fail_on_status_code=False,
    )
    assert response.status == 400, (
        f"Expected 400 for gibberish profile, got {response.status}"
    )


@pytest.mark.neg
@pytest.mark.backend
def test_neg_a02_random_ascii_profile_returns_400(backend_api: APIRequestContext):
    """
    TC-NEG-A02 – 60-character random ASCII noise as profile name.

    Input  : Valid records, profile = random printable ASCII (60 chars)
    Expect : HTTP 400.
    """
    gibberish_profile = _random_string(60)
    response = backend_api.post(
        "/process",
        data={"records": VALID_RECORDS, "profile": gibberish_profile},
        fail_on_status_code=False,
    )
    assert response.status == 400, (
        f"Gibberish profile '{gibberish_profile[:20]}…' should return 400, got {response.status}"
    )


@pytest.mark.neg
@pytest.mark.backend
def test_neg_a03_empty_string_profile_returns_400(backend_api: APIRequestContext):
    """
    TC-NEG-A03 – Empty string as profile name.

    Input  : Valid records, profile = ''
    Expect : HTTP 400 (empty profile is not a known profile).
    """
    response = backend_api.post(
        "/process",
        data={"records": VALID_RECORDS, "profile": ""},
        fail_on_status_code=False,
    )
    assert response.status in (400, 422), (
        f"Empty profile should return 400/422, got {response.status}"
    )


@pytest.mark.neg
@pytest.mark.backend
def test_neg_a04_none_profile_returns_422(backend_api: APIRequestContext):
    """
    TC-NEG-A04 – Missing 'profile' key entirely.

    Input  : Valid records, no profile field
    Expect : HTTP 422 (Pydantic rejects missing required field).
    """
    response = backend_api.post(
        "/process",
        data={"records": VALID_RECORDS},
        fail_on_status_code=False,
    )
    assert response.status == 422, f"Expected 422 for missing profile, got {response.status}"


@pytest.mark.neg
@pytest.mark.backend
def test_neg_a05_records_as_plain_string_returns_422(backend_api: APIRequestContext):
    """
    TC-NEG-A05 – 'records' field is a string, not a list.

    Input  : records = "this is not a list", profile = valid
    Expect : HTTP 422.
    """
    response = backend_api.post(
        "/process",
        data={"records": "this is not a list at all !!!", "profile": VALID_PROFILE},
        fail_on_status_code=False,
    )
    assert response.status == 422, f"Expected 422 for string records, got {response.status}"


@pytest.mark.neg
@pytest.mark.backend
def test_neg_a06_store_with_numeric_table_name(backend_api: APIRequestContext):
    """
    TC-NEG-A06 – Table name that is just numbers.

    Input  : table = '12345', valid records
    Expect : 200 or 400 – must NOT be 500.
    """
    response = backend_api.post(
        "/store",
        data={"table": "12345", "records": [{"x": 1}]},
        fail_on_status_code=False,
    )
    assert response.status != 500, (
        f"Numeric table name caused server crash (500): {response.text()}"
    )


@pytest.mark.neg
@pytest.mark.backend
def test_neg_a07_validate_with_unknown_dtype_does_not_crash(backend_api: APIRequestContext):
    """
    TC-NEG-A07 – Validation rule with unknown dtype value.

    Input  : dtype = 'gibberish_type'
    Expect : 200 (unknown dtype is treated as no-type-check) OR 500 must not occur.
    """
    response = backend_api.post(
        "/validate",
        data={
            "records": VALID_RECORDS,
            "rules": [{"name": "Company Name", "dtype": "gibberish_type", "required": True}],
        },
        fail_on_status_code=False,
    )
    assert response.status != 500, (
        f"Gibberish dtype caused server crash: {response.text()}"
    )


@pytest.mark.neg
@pytest.mark.backend
def test_neg_a08_process_with_unicode_noise_in_records(backend_api: APIRequestContext):
    """
    TC-NEG-A08 – Record values contain emoji and mixed Unicode noise.

    Input  : Company Name = '🎲🔥💥 INVALID ⚡🌪️', Revenue = -9999.99
    Expect : Service processes (200) or rejects (400/422), must not crash (500).
    """
    unicode_records = [
        {
            "Company Name": "🎲🔥💥 ẤẼẾ INVALID ⚡🌪️ Ñoño",
            "Revenue": -9999.99,
            "Region": "Ñorth",
            "Report Date": "9999-99-99",
        }
    ]
    response = backend_api.post(
        "/process",
        data={"records": unicode_records, "profile": VALID_PROFILE},
        fail_on_status_code=False,
    )
    assert response.status != 500, (
        f"Unicode noise caused server crash (500): {response.text()}"
    )


@pytest.mark.neg
@pytest.mark.backend
def test_neg_a09_extreme_long_string_as_profile(backend_api: APIRequestContext):
    """
    TC-NEG-A09 – Profile name is an extremely long string (5000 chars).

    Input  : profile = 'X' * 5000
    Expect : HTTP 400, must not cause 500 or timeout crash.
    """
    very_long = "XGIBBERISH" * 500   # 5000 chars
    response = backend_api.post(
        "/process",
        data={"records": VALID_RECORDS, "profile": very_long},
        fail_on_status_code=False,
    )
    assert response.status in (400, 422), (
        f"5000-char profile caused {response.status}: {response.text()[:100]}"
    )


@pytest.mark.neg
@pytest.mark.backend
def test_neg_a10_retrieve_with_special_chars_table_name(backend_api: APIRequestContext):
    """
    TC-NEG-A10 – Retrieve from table name containing special characters in URL.

    Input  : GET /retrieve/table@#!%^
    Expect : 404 or 422 – must not return 500.
    """
    response = backend_api.get("/retrieve/table_gibberish_xyz_!@#", fail_on_status_code=False)
    assert response.status in (400, 404, 422), (
        f"Special-char table name in URL caused {response.status}"
    )


# ===========================================================================
# SECTION B – SQL Injection & Security inputs (all should PASS)
# ===========================================================================

@pytest.mark.neg
@pytest.mark.backend
def test_neg_b01_sql_injection_in_profile_returns_400(backend_api: APIRequestContext):
    """
    TC-NEG-B01 – Classic SQL injection in the profile field.

    Input  : profile = "'; DROP TABLE companies; --"
    Expect : HTTP 400 (unknown profile) – server does NOT execute SQL.
    """
    sql_injection = "'; DROP TABLE companies; -- "
    response = backend_api.post(
        "/process",
        data={"records": VALID_RECORDS, "profile": sql_injection},
        fail_on_status_code=False,
    )
    assert response.status == 400, (
        f"SQL injection in profile returned {response.status} instead of 400"
    )


@pytest.mark.neg
@pytest.mark.backend
def test_neg_b02_sql_injection_in_record_value_handled_safely(backend_api: APIRequestContext):
    """
    TC-NEG-B02 – SQL injection string inside a record value.

    Input  : Company Name = "'; DELETE FROM users; --"
    Expect : Service handles it (200/400), must NOT return 500 or crash.
    """
    injected_records = [
        {
            "Company Name": "'; DELETE FROM users; --",
            "Revenue": 100.0,
            "Region": "South",
            "Report Date": "2026-01-01",
        }
    ]
    response = backend_api.post(
        "/process",
        data={"records": injected_records, "profile": VALID_PROFILE},
        fail_on_status_code=False,
    )
    assert response.status != 500, (
        f"SQL injection in record value caused server crash: {response.text()}"
    )


@pytest.mark.neg
@pytest.mark.backend
def test_neg_b03_xss_payload_in_record_does_not_crash_server(backend_api: APIRequestContext):
    """
    TC-NEG-B03 – XSS script tag inside a record string field.

    Input  : Company Name = '<script>alert("xss")</script>'
    Expect : 200 or 400 – server does NOT execute JS, must not 500.
    """
    xss_records = [
        {
            "Company Name": '<script>alert("xss")</script>',
            "Revenue": 200.0,
            "Region": "East",
            "Report Date": "2026-01-01",
        }
    ]
    response = backend_api.post(
        "/process",
        data={"records": xss_records, "profile": VALID_PROFILE},
        fail_on_status_code=False,
    )
    assert response.status != 500, (
        f"XSS payload in record caused crash: {response.text()}"
    )


@pytest.mark.neg
@pytest.mark.backend
def test_neg_b04_null_byte_injection_in_profile(backend_api: APIRequestContext):
    """
    TC-NEG-B04 – Null byte in profile string.

    Input  : profile = "company_financials\x00malicious"
    Expect : 400 – server rejects or truncates at null byte safely.
    """
    null_profile = "company_financials\x00malicious_suffix"
    response = backend_api.post(
        "/process",
        data={"records": VALID_RECORDS, "profile": null_profile},
        fail_on_status_code=False,
    )
    # Either 400 (rejected) or 200 (truncated to valid profile) – must not be 500
    assert response.status != 500, (
        f"Null byte injection caused server crash: {response.text()}"
    )


@pytest.mark.neg
@pytest.mark.backend
def test_neg_b05_path_traversal_in_table_name(backend_api: APIRequestContext):
    """
    TC-NEG-B05 – Path traversal attempt in table name via /retrieve.

    Input  : GET /retrieve/../../etc/passwd
    Expect : 404 or 400 – server does NOT traverse paths.
    """
    response = backend_api.get(
        "/retrieve/..%2F..%2Fetc%2Fpasswd", fail_on_status_code=False
    )
    assert response.status in (400, 404, 422), (
        f"Path traversal attempt returned unexpected {response.status}"
    )


# ===========================================================================
# SECTION C – Intentional XFAIL tests  (marked @pytest.mark.xfail)
# Purpose : demonstrate failure-detection; appear as XFAIL in report
# ===========================================================================

@pytest.mark.neg
@pytest.mark.backend
@pytest.mark.xfail(
    strict=True,
    reason="INTENTIONAL XFAIL TC-NEG-C01: Health endpoint returns 200 not 201 – "
           "wrong status asserted deliberately to demonstrate failure detection.",
)
def test_neg_c01_xfail_health_wrongly_expects_201(backend_api: APIRequestContext):
    """
    TC-NEG-C01 [INTENTIONAL XFAIL] – Asserts health returns 201.

    Why: Health always returns 200. Asserting 201 is deliberately wrong.
    Evidence value: Proves the test framework detects wrong status assertions.
    """
    response = backend_api.get("/health")
    assert response.status == 201, (
        f"INTENTIONAL: expected 201, actual {response.status} (always 200)"
    )


@pytest.mark.neg
@pytest.mark.backend
@pytest.mark.xfail(
    strict=True,
    reason="INTENTIONAL XFAIL TC-NEG-C02: /sources response does not contain "
           "'xml_sources' key – asserting non-existent field deliberately.",
)
def test_neg_c02_xfail_sources_wrongly_expects_xml_sources(backend_api: APIRequestContext):
    """
    TC-NEG-C02 [INTENTIONAL XFAIL] – Asserts /sources has 'xml_sources' key.

    Why: The backend only exposes 'csv_sources'. Asserting 'xml_sources' is wrong.
    Evidence value: Confirms the suite catches missing-field regressions.
    """
    body = backend_api.get("/sources").json()
    assert "xml_sources" in body, (
        f"INTENTIONAL: 'xml_sources' does not exist in response keys: {list(body.keys())}"
    )


@pytest.mark.neg
@pytest.mark.backend
@pytest.mark.xfail(
    strict=True,
    reason="INTENTIONAL XFAIL TC-NEG-C03: Process with empty records returns 400, "
           "not 200 – asserting wrong status code deliberately.",
)
def test_neg_c03_xfail_empty_records_wrongly_expects_200(backend_api: APIRequestContext):
    """
    TC-NEG-C03 [INTENTIONAL XFAIL] – Sends empty records and expects 200.

    Why: Empty records correctly return 400. Asserting 200 is deliberately wrong.
    Evidence value: Confirms input validation is enforced.
    """
    response = backend_api.post(
        "/process",
        data={"records": [], "profile": VALID_PROFILE},
        fail_on_status_code=False,
    )
    assert response.status == 200, (
        f"INTENTIONAL: expected 200 for empty records, actual {response.status}"
    )


@pytest.mark.neg
@pytest.mark.backend
@pytest.mark.xfail(
    strict=True,
    reason="INTENTIONAL XFAIL TC-NEG-C04: Validate with empty list returns 400, "
           "not 201 – deliberately wrong assertion.",
)
def test_neg_c04_xfail_validate_empty_wrongly_expects_201(backend_api: APIRequestContext):
    """
    TC-NEG-C04 [INTENTIONAL XFAIL] – Validates empty records, asserts 201.

    Why: /validate with empty records returns 400. Asserting 201 is wrong.
    """
    response = backend_api.post(
        "/validate",
        data={
            "records": [],
            "rules": [{"name": "col", "required": True}],
        },
        fail_on_status_code=False,
    )
    assert response.status == 201, (
        f"INTENTIONAL: expected 201, actual {response.status}"
    )


@pytest.mark.neg
@pytest.mark.backend
@pytest.mark.xfail(
    strict=True,
    reason="INTENTIONAL XFAIL TC-NEG-C05: /retrieve on nonexistent table returns 404, "
           "not 200 – deliberate wrong-status assertion.",
)
def test_neg_c05_xfail_nonexistent_table_wrongly_expects_200(backend_api: APIRequestContext):
    """
    TC-NEG-C05 [INTENTIONAL XFAIL] – Retrieves from nonexistent table and expects 200.

    Why: Nonexistent table returns 404. Asserting 200 is deliberately wrong.
    """
    response = backend_api.get(
        "/retrieve/absolutely_nonexistent_table_zzz9999",
        fail_on_status_code=False,
    )
    assert response.status == 200, (
        f"INTENTIONAL: expected 200 for nonexistent table, actual {response.status}"
    )


# ===========================================================================
# SECTION D – Intended FAIL tests (marked xfail so CI passes with expected failure)
# Purpose : Demonstrate what a real regression failure looks like in the report
# ===========================================================================

@pytest.mark.neg
@pytest.mark.backend
@pytest.mark.xfail(
    strict=True,
    reason="[INTENDED FAIL] TC-NEG-D01: Asserts status='running' instead of 'ok' to demonstrate failure detection.",
)
def test_neg_d01_genuine_fail_health_body_wrong_status_value(backend_api: APIRequestContext):
    """
    TC-NEG-D01 [INTENDED FAIL] – Asserts health body has status='running'.

    Why: Health body actually contains status='ok'.
    Demonstrates: How a bug in status-field naming would appear in CI.
    """
    body = backend_api.get("/health").json()
    # DELIBERATE WRONG ASSERTION – actual value is 'ok', not 'running'
    assert body.get("status") == "running", (
        f"[INTENDED FAIL] Expected status='running', got status='{body.get('status')}'"
    )


@pytest.mark.neg
@pytest.mark.backend
@pytest.mark.xfail(
    strict=True,
    reason="[INTENDED FAIL] TC-NEG-D02: Asserts 'net_profit' column in response to demonstrate failure detection.",
)
def test_neg_d02_genuine_fail_process_wrong_column_in_response(backend_api: APIRequestContext):
    """
    TC-NEG-D02 [INTENDED FAIL] – Asserts processed data has column 'net_profit'.

    Why: The company_financials profile does not produce a 'net_profit' column.
    Demonstrates: How a missing KPI column would fail in regression.
    """
    response = backend_api.post(
        "/process",
        data={"records": VALID_RECORDS, "profile": VALID_PROFILE},
    )
    assert response.status == 200
    columns = response.json().get("columns", [])
    # DELIBERATE WRONG ASSERTION – 'net_profit' is not produced by this profile
    assert "net_profit" in columns, (
        f"[INTENDED FAIL] Expected 'net_profit' column, actual columns: {columns}"
    )


@pytest.mark.neg
@pytest.mark.backend
@pytest.mark.xfail(
    strict=True,
    reason="[INTENDED FAIL] TC-NEG-D03: Asserts rows_written=99 instead of 1 to demonstrate failure detection.",
)
def test_neg_d03_genuine_fail_store_expects_wrong_rows_written(backend_api: APIRequestContext):
    """
    TC-NEG-D03 [INTENDED FAIL] – Stores 1 record but asserts 99 were written.

    Why: Only 1 record is provided, so rows_written == 1, not 99.
    Demonstrates: Regression detection for write-count verification.
    """
    response = backend_api.post(
        "/store",
        data={"table": "carivix_neg_d03", "records": [{"x": 42}]},
    )
    assert response.status == 200
    rows = response.json().get("rows_written", 0)
    # DELIBERATE WRONG ASSERTION
    assert rows == 99, (
        f"[INTENDED FAIL] Expected rows_written=99, actual {rows}"
    )
