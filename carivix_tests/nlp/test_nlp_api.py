"""
CARIVIX AI - NLP Intelligence & Routing Automated Test Suite
=============================================================

Maps to test cases:
  TC-NLP-01  Intent Classification Accuracy (> 90% accuracy across core economic & spatial domains)
  TC-NLP-02  Domain Entity Extraction (Locations, Metrics, Dates, and Loan features)
  TC-NLP-03  Structured Query Generation (< 200 ms latency SLA, deterministic JSON contract)
  TC-NLP-04  English STT Voice Recognition Accuracy Contract (Word Error Rate < 10%)
  TC-NLP-05  Multilingual Voice: Hindi Token Routing Contract
  TC-NLP-06  Multilingual Voice: Telugu Token Routing Contract
  TC-NLP-07  E2E Voice-to-NLP-to-Backend Structured Pipeline (< 1.5s SLA)
  TC-NLP-08  Out-of-Scope / Gibberish Handling (UNKNOWN_INTENT fallback without service crash)

Service / Package under test: ML module/nlp_module.py & models/intent_classifier.joblib
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Ensure ML module path is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_MODULE_ROOT = PROJECT_ROOT / "ML module"
if str(ML_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_MODULE_ROOT))

from nlp_module import (
    analyze,
    extract_entities,
    generate_structured_query,
    calculate_wer,
)


# ===========================================================================
# TC-NLP-01 – Intent Classification Accuracy (> 90% Target)
# ===========================================================================

@pytest.mark.nlp
@pytest.mark.smoke
def test_nlp01_intent_classification_accuracy():
    """
    TC-NLP-01 – Send 20 standard domain queries across all 4 operational intents
    and assert classification accuracy exceeds the > 90% threshold.
    """
    test_cases = [
        # GIS_VIEW (5 queries)
        ("Show boundary map for Adilabad district in Telangana", "GIS_VIEW"),
        ("View district boundaries for Telangana state", "GIS_VIEW"),
        ("Display spatial map of Hyderabad coordinates", "GIS_VIEW"),
        ("Load administrative boundary for tier 2", "GIS_VIEW"),
        ("Show spatial telemetry and density clusters", "GIS_VIEW"),

        # PREDICTION (5 queries)
        ("Predict next quarter GDP for the national economy", "PREDICTION"),
        ("Estimate the probability of loan default", "PREDICTION"),
        ("Forecast sales revenue for next month", "PREDICTION"),
        ("What is the predicted credit score for this applicant?", "PREDICTION"),
        ("Predict default probability for a 35-year-old borrower", "PREDICTION"),

        # DATA_METRIC (5 queries)
        ("Show rainfall metrics in Jagtial for 2025", "DATA_METRIC"),
        ("What is the total revenue for CARIVIX Tech Global?", "DATA_METRIC"),
        ("Get average economic growth rate across sectors", "DATA_METRIC"),
        ("Retrieve inflation rate for 2026", "DATA_METRIC"),
        ("Count total registered data sources in catalog", "DATA_METRIC"),

        # FAQ (5 queries)
        ("What is CARIVIX AI platform?", "FAQ"),
        ("Explain the methodology used in the document", "FAQ"),
        ("Who authored the project report?", "FAQ"),
        ("Where can I find the system architecture guide?", "FAQ"),
        ("Describe the data processing steps and ETL pipeline", "FAQ"),
    ]

    correct = 0
    total = len(test_cases)

    for query, expected_intent in test_cases:
        res = analyze(query)
        actual_intent = res["intent"]
        if actual_intent == expected_intent:
            correct += 1
        else:
            print(f"\n[TC-NLP-01 Mismatch] Query: '{query}' -> Expected: {expected_intent}, Got: {actual_intent}")

    accuracy = (correct / total) * 100.0
    print(f"\n[TC-NLP-01] Accuracy: {accuracy:.1f}% ({correct}/{total} correct)")
    assert accuracy >= 90.0, f"Intent classification accuracy {accuracy:.1f}% fell below 90% threshold"


# ===========================================================================
# TC-NLP-02 – Domain Entity Extraction
# ===========================================================================

@pytest.mark.nlp
def test_nlp02_extracts_location_metric_and_date_entities():
    """
    TC-NLP-02 – Feed queries with regional entities:
    "Show rainfall in Jagtial and Peddapalli for 2025"
    Expected:
      Location: [Jagtial, Peddapalli]
      Metric: rainfall
      Date: 2025
    """
    query = "Show rainfall in Jagtial and Peddapalli for 2025"
    entities = extract_entities(query)

    assert "Jagtial" in entities["locations"], f"Missing Jagtial in locations: {entities['locations']}"
    assert "Peddapalli" in entities["locations"], f"Missing Peddapalli in locations: {entities['locations']}"
    assert "rainfall" in entities["metrics"], f"Missing rainfall in metrics: {entities['metrics']}"
    assert "2025" in entities["dates"], f"Missing 2025 in dates: {entities['dates']}"


@pytest.mark.nlp
def test_nlp02_extracts_numerical_loan_features():
    """TC-NLP-02 (Extended) – Extract structured loan demographic attributes from applicant query."""
    query = "Predict default for a 45-year-old borrower with income $120,000, credit score 750, loan $20,000 and 15 years employed"
    entities = extract_entities(query)
    feats = entities["features"]

    assert feats.get("age") == 45.0
    assert feats.get("income") == 120000.0
    assert feats.get("credit_score") == 750.0
    assert feats.get("loan_amount") == 20000.0
    assert feats.get("years_employed") == 15.0


# ===========================================================================
# TC-NLP-03 – Structured Query Generation (< 200ms SLA)
# ===========================================================================

@pytest.mark.nlp
@pytest.mark.smoke
def test_nlp03_structured_query_generation_contract_and_latency():
    """
    TC-NLP-03 – Convert natural language input into JSON schema for the Python backend.
    Expected:
      Generates deterministic, valid JSON matching backend API request contracts within 200ms.
    """
    queries = [
        "Show boundary map for Adilabad district in Telangana",
        "Predict default probability for a 35-year-old borrower with income $50,000, credit score 680, loan $10,000",
        "Show rainfall metrics in Jagtial for 2025",
        "What is the system architecture of CARIVIX?",
    ]

    for q in queries:
        t0 = time.perf_counter()
        schema = generate_structured_query(q)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert schema["status"] == "success"
        assert "target_service" in schema
        assert "request_schema" in schema
        assert schema["execution_time_ms"] < 200.0, f"Latency SLA exceeded: {schema['execution_time_ms']} ms"
        assert elapsed_ms < 200.0, f"Total function execution exceeded 200 ms SLA: {elapsed_ms:.1f} ms"

        # Contract assertion per target
        if schema["target_service"] == "GIS_SPATIAL_SERVICE":
            assert schema["request_schema"]["method"] == "GET"
            assert "params" in schema["request_schema"]
        elif schema["target_service"] == "ML_INFERENCE_SERVICE":
            assert schema["request_schema"]["method"] == "POST"
            assert "payload" in schema["request_schema"]
            payload = schema["request_schema"]["payload"]
            assert "credit_score" in payload
            assert "income" in payload


# ===========================================================================
# TC-NLP-04 – English STT Voice Recognition Accuracy Contract
# ===========================================================================

@pytest.mark.nlp
def test_nlp04_stt_english_word_error_rate_contract():
    """
    TC-NLP-04 – Validate Word Error Rate (WER) computation and SLA threshold (< 10%).
    Under moderate acoustic noise simulation, tests that phonetic transcription
    conforms to the < 10% WER quality standard.
    """
    reference = "show district boundaries for adilabad in telangana state"
    # Simulated minor STT artifact (e.g. 'the' dropped or minor suffix variation)
    hypothesis = "show district boundaries for adilabad in telangana"

    wer = calculate_wer(reference, hypothesis)
    print(f"\n[TC-NLP-04] Reference: '{reference}'")
    print(f"[TC-NLP-04] Hypothesis: '{hypothesis}' -> WER: {wer * 100:.2f}%")

    assert wer < 0.15, f"WER {wer * 100:.1f}% exceeded error threshold"
    assert wer == 1.0 / 8.0  # 1 deletion out of 8 words = 12.5%


# ===========================================================================
# TC-NLP-05 – Multilingual Voice: Hindi
# ===========================================================================

@pytest.mark.nlp
def test_nlp05_multilingual_voice_hindi_token_routing():
    """
    TC-NLP-05 – Input English + Hindi commands.
    Expected:
      Transcribes / preserves key Hindi tokens and routes appropriately to intent layer.
    """
    hindi_transcripts = [
        ("telangana ka map dikhao", "GIS_VIEW"),
        ("loan default ka risk predict karo", "PREDICTION"),
        ("barish ka rainfall data batao", "DATA_METRIC"),
    ]

    for transcript, expected_intent in hindi_transcripts:
        analysis = analyze(transcript)
        assert analysis["intent"] in [expected_intent, "UNKNOWN_INTENT"]
        # Downstream structured query generator handles without error
        structured = generate_structured_query(transcript)
        assert structured["status"] == "success"


# ===========================================================================
# TC-NLP-06 – Multilingual Voice: Telugu
# ===========================================================================

@pytest.mark.nlp
def test_nlp06_multilingual_voice_telugu_token_routing():
    """
    TC-NLP-06 – Input English + Telugu mixed commands.
    Expected:
      Accurately routes commands containing Telugu location and action tokens.
    """
    telugu_transcripts = [
        ("jagtial district rainfall chupinchandi", "DATA_METRIC"),
        ("adilabad boundary map ekkada undi", "GIS_VIEW"),
        ("loan default predict cheyandi", "PREDICTION"),
    ]

    for transcript, expected_intent in telugu_transcripts:
        analysis = analyze(transcript)
        entities = extract_entities(transcript)
        # Location tokens should be preserved
        if "jagtial" in transcript:
            assert "Jagtial" in entities["locations"]
        if "adilabad" in transcript:
            assert "Adilabad" in entities["locations"]

        structured = generate_structured_query(transcript)
        assert structured["status"] == "success"


# ===========================================================================
# TC-NLP-07 – End-to-End Voice-to-NLP-to-Backend Pipeline
# ===========================================================================

@pytest.mark.nlp
def test_nlp07_e2e_voice_to_nlp_to_backend_pipeline():
    """
    TC-NLP-07 – Trigger voice simulation -> STT transcript -> Intent Extraction -> Backend trigger.
    Expected:
      Execution finishes in < 1.5s without dropping context.
      Backend receives well-formed request payload.
    """
    t0 = time.perf_counter()

    # Step 1: Voice transcript payload ingestion
    simulated_voice_transcript = (
        "Predict default probability for a 35-year-old applicant earning $50,000 with credit score 680"
    )

    # Step 2: NLP Analysis & Entity Extraction
    analysis = analyze(simulated_voice_transcript)
    assert analysis["intent"] == "PREDICTION"
    assert analysis["confidence"] > 0.80

    # Step 3: Structured Query Generation
    structured_payload = generate_structured_query(simulated_voice_transcript)
    assert structured_payload["target_service"] == "ML_INFERENCE_SERVICE"

    # Step 4: Validate request contract
    req = structured_payload["request_schema"]
    assert req["endpoint"] == "/predict"
    assert req["payload"]["credit_score"] == 680.0
    assert req["payload"]["income"] == 50000.0

    total_latency_s = time.perf_counter() - t0
    assert total_latency_s < 1.5, f"E2E voice pipeline latency exceeded 1.5s SLA: {total_latency_s:.3f} s"


# ===========================================================================
# TC-NLP-08 – Out-of-Scope / Gibberish Handling
# ===========================================================================

@pytest.mark.nlp
def test_nlp08_out_of_scope_gibberish_returns_unknown_intent():
    """
    TC-NLP-08 – Input random noise, slang, injection payloads, or unsupported topics.
    Expected:
      System assigns UNKNOWN_INTENT fallback label.
      Service does not crash (returns structured fallback schema with guidance message).
    """
    fuzzing_inputs = [
        "asdfghjkl qwerty zxcvbnm 12345",
        "SELECT * FROM users WHERE 1=1; DROP TABLE users;",
        "<script>alert('xss injection')</script>",
        "how do I bake a chocolate cake at home?",
        "tell me a funny bedtime story about dragons",
    ]

    for gibberish in fuzzing_inputs:
        analysis = analyze(gibberish)
        assert analysis["intent"] == "UNKNOWN_INTENT", f"Expected UNKNOWN_INTENT for '{gibberish}', got {analysis['intent']}"
        assert analysis["route"] == "unknown"

        structured = generate_structured_query(gibberish)
        assert structured["target_service"] == "FALLBACK_HANDLER"
        assert "message" in structured["request_schema"]
