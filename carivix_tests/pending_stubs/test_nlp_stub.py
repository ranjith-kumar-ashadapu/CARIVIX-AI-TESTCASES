"""
NLP Team – Test Stubs
======================

Status  : PENDING – NLP team delivery
TC-IDs  : TC-NLP-01 .. TC-NLP-08

These tests are marked @pytest.mark.skip.
When the NLP team delivers their service / module:
  1. Remove the @pytest.mark.skip decorator from the relevant function.
  2. Fill in the test body.
  3. Move the function to an active test file or rename this file to
     test_nlp_api.py and activate it.

Run stubs to confirm they appear as SKIPPED:
    pytest carivix_tests/pending_stubs/test_nlp_stub.py -v
"""

import pytest

_REASON = "PENDING – NLP team delivery: service/module not yet available"


# ===========================================================================
# TC-NLP-01 – Speech-to-Text Accuracy: English
# ===========================================================================

@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_nlp01_stt_english_transcription_accuracy():
    """
    TC-NLP-01 – Feed a pre-recorded English audio clip to the STT module.

    Expected:
        Word Error Rate (WER) < 5% on the reference transcript.
        Transcription latency < 500 ms for a 5-second clip.

    TODO:
        1. Load test audio file from NLP team's test data directory.
        2. Call the STT endpoint (URL TBD) with the audio payload.
        3. Assert WER against the reference transcript.
    """
    raise NotImplementedError


# ===========================================================================
# TC-NLP-02 – Intent Classification: Core Queries
# ===========================================================================

@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_nlp02_intent_classification_core_economic_queries():
    """
    TC-NLP-02 – Send 15 standard economic-domain queries to the intent classifier.

    Expected:
        All 15 are mapped to the correct intent label.
        Zero unknown-intent fallbacks for in-scope queries.
        Confidence score > 0.85 for each classification.

    TODO:
        1. Define the 15 test queries and their expected intents.
        2. POST each query to the NLP intent endpoint (URL TBD).
        3. Assert intent label and confidence threshold.
    """
    raise NotImplementedError


# ===========================================================================
# TC-NLP-03 – Named Entity Recognition (NER)
# ===========================================================================

@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_nlp03_ner_extracts_location_and_metric_entities():
    """
    TC-NLP-03 – Input query mentioning districts, KPIs, and time periods.

    Expected:
        NER module extracts LOCATION, KPI, and TIME_PERIOD entities correctly.
        No spurious entities for unrelated tokens.

    TODO:
        1. Define test queries with known entity spans.
        2. POST to NER endpoint (URL TBD).
        3. Assert extracted entities match expected spans.
    """
    raise NotImplementedError


# ===========================================================================
# TC-NLP-04 – Contextual Query Disambiguation
# ===========================================================================

@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_nlp04_contextual_query_disambiguation():
    """
    TC-NLP-04 – Submit ambiguous query with prior context (multi-turn).

    Expected:
        System resolves pronouns using the conversation history.
        Response includes the disambiguated entity reference.

    TODO:
        1. Build a multi-turn conversation payload (context + new query).
        2. POST to the NLP disambiguation endpoint (URL TBD).
        3. Assert that the resolved entity is correct.
    """
    raise NotImplementedError


# ===========================================================================
# TC-NLP-05 – Multilingual Voice: Hindi
# ===========================================================================

@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_nlp05_stt_hindi_multilingual_accuracy():
    """
    TC-NLP-05 – Input spoken Hindi commands via audio.

    Expected:
        Audio accurately transcribes Hindi tokens and routes correctly.
        No code-switch errors for pure Hindi input.

    TODO:
        1. Load Hindi audio test file.
        2. Call STT endpoint with language='hi' header (URL TBD).
        3. Assert WER < 10% for Hindi.
    """
    raise NotImplementedError


# ===========================================================================
# TC-NLP-06 – Multilingual Voice: Telugu
# ===========================================================================

@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_nlp06_stt_telugu_multilingual_accuracy():
    """
    TC-NLP-06 – Input mixed English + Telugu spoken commands.

    Expected:
        Audio accurately transcribes both English and Telugu tokens.
        Routes to intent classification layer without dropping tokens.

    TODO:
        1. Load Telugu/mixed audio test file.
        2. Call STT endpoint with appropriate language config.
        3. Assert key Telugu tokens are present in the transcript.
    """
    raise NotImplementedError


# ===========================================================================
# TC-NLP-07 – End-to-End Voice-to-NLP-to-Backend Pipeline
# ===========================================================================

@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_nlp07_e2e_voice_to_nlp_to_backend_pipeline():
    """
    TC-NLP-07 – Trigger voice command → STT → Intent Extraction → Backend trigger.

    Expected:
        Complete voice-to-structured-intent execution finishes in < 1.5 s.
        No context is dropped between pipeline stages.
        Backend receives a well-formed intent payload.

    TODO:
        1. Simulate voice input via the NLP gateway (URL TBD).
        2. Assert each stage completes within its sub-latency budget.
        3. Poll backend for the triggered intent record.
    """
    raise NotImplementedError


# ===========================================================================
# TC-NLP-08 – Out-of-Scope / Gibberish Handling
# ===========================================================================

@pytest.mark.pending
@pytest.mark.skip(reason=_REASON)
def test_nlp08_out_of_scope_gibberish_returns_unknown_intent():
    """
    TC-NLP-08 – Input random noise, slang, or unsupported topics.

    Expected:
        System assigns UNKNOWN_INTENT fallback label.
        A polite re-prompt response is generated.
        Service does NOT crash (HTTP 200 with fallback, not 500).

    TODO:
        1. POST nonsensical queries to the NLP intent endpoint.
        2. Assert intent == "UNKNOWN_INTENT".
        3. Assert response status is 200 (not 5xx).
    """
    raise NotImplementedError
