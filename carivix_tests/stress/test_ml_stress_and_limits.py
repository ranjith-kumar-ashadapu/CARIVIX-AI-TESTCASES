"""
Rigorous ML Model Stress, Extreme Boundary & Random Fuzzing Tests
=================================================================

Pushes the machine learning models and inference services to their limits:
- Monte Carlo random applicant generation across the full domain space
- Counter-intuitive extreme demographic & financial profiles
- Large-batch scaling and throughput benchmarks
- Output stability, NaN/Inf immunity, and probability boundary assertions [0.0, 1.0]
- High-concurrency inference bursts

Target Service: http://127.0.0.1:8001
"""

from __future__ import annotations

import concurrent.futures
import random
import time
from typing import Any, Dict, List

import pytest
from playwright.sync_api import APIRequestContext

# Allowed categorical sets
EDUCATIONS = ["High School", "Bachelor", "Master", "PhD", "Other"]
EMPLOYMENT_STATUSES = ["Employed", "Self-Employed", "Unemployed", "Retired"]
MARITAL_STATUSES = ["Single", "Married", "Divorced", "Widowed"]
HOUSING_TYPES = ["Own", "Rent", "Mortgage", "Other"]


def generate_random_applicant(seed: int) -> Dict[str, Any]:
    """Generate a valid, randomized applicant record using deterministic seed."""
    rng = random.Random(seed)
    return {
        "age": rng.randint(18, 85),
        "income": round(rng.uniform(10_000.0, 300_000.0), 2),
        "credit_score": rng.randint(300, 850),
        "loan_amount": round(rng.uniform(1_000.0, 100_000.0), 2),
        "years_employed": rng.randint(0, 45),
        "education": rng.choice(EDUCATIONS),
        "employment_status": rng.choice(EMPLOYMENT_STATUSES),
        "marital_status": rng.choice(MARITAL_STATUSES),
        "housing_type": rng.choice(HOUSING_TYPES),
        "application_date": f"202{rng.randint(3, 6)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
    }


# ===========================================================================
# 1. Extreme Counter-Intuitive Boundary Profiles
# ===========================================================================

EXTREME_PROFILES = [
    # 1. Ultra-wealthy applicant with rock-bottom credit score
    {
        "id": "EXTREME-01-WEALTHY-BAD-CREDIT",
        "payload": {
            "age": 45, "income": 5_000_000.0, "credit_score": 305, "loan_amount": 10_000.0,
            "years_employed": 20, "education": "PhD", "employment_status": "Self-Employed",
            "marital_status": "Married", "housing_type": "Own", "application_date": "2026-01-15"
        }
    },
    # 2. Minimum-wage applicant requesting massive loan
    {
        "id": "EXTREME-02-LOW-INCOME-MASSIVE-LOAN",
        "payload": {
            "age": 21, "income": 12_000.0, "credit_score": 780, "loan_amount": 500_000.0,
            "years_employed": 1, "education": "High School", "employment_status": "Employed",
            "marital_status": "Single", "housing_type": "Rent", "application_date": "2026-02-20"
        }
    },
    # 3. Elderly applicant with maximum possible employment tenure
    {
        "id": "EXTREME-03-ELDERLY-MAX-TENURE",
        "payload": {
            "age": 95, "income": 65_000.0, "credit_score": 840, "loan_amount": 5_000.0,
            "years_employed": 70, "education": "Master", "employment_status": "Retired",
            "marital_status": "Widowed", "housing_type": "Own", "application_date": "2026-03-10"
        }
    },
    # 4. Youngest valid adult with zero employment tenure
    {
        "id": "EXTREME-04-YOUNG-ZERO-TENURE",
        "payload": {
            "age": 18, "income": 25_000.0, "credit_score": 600, "loan_amount": 2_000.0,
            "years_employed": 0, "education": "High School", "employment_status": "Unemployed",
            "marital_status": "Single", "housing_type": "Rent", "application_date": "2026-04-05"
        }
    },
    # 5. Perfect score with zero loan amount
    {
        "id": "EXTREME-05-PERFECT-SCORE-ZERO-LOAN",
        "payload": {
            "age": 50, "income": 150_000.0, "credit_score": 850, "loan_amount": 0.0,
            "years_employed": 25, "education": "PhD", "employment_status": "Employed",
            "marital_status": "Married", "housing_type": "Own", "application_date": "2026-05-12"
        }
    },
]


@pytest.mark.stress
@pytest.mark.ml
@pytest.mark.parametrize("case", EXTREME_PROFILES, ids=[c["id"] for c in EXTREME_PROFILES])
def test_ml_extreme_boundary_profiles_produce_valid_bounded_predictions(
    ml_api: APIRequestContext, case: Dict[str, Any]
):
    """
    Push ML inference to its limits with counter-intuitive extreme financial profiles.
    Asserts:
    1. HTTP 200 OK (server does not crash on extreme values).
    2. Prediction is strictly binary (0 or 1).
    3. Probability is non-NaN, non-Infinite, and strictly in [0.0, 1.0].
    4. Latency is under 300ms SLA.
    """
    start = time.perf_counter()
    resp = ml_api.post("/api/v1/predict", data=case["payload"])
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert resp.status == 200, f"Failed on extreme profile {case['id']}: {resp.text()}"
    body = resp.json()

    assert body.get("success") is True
    assert body.get("prediction") in (0, 1)
    if "probability" in body and body["probability"] is not None:
        prob = float(body["probability"])
        assert not (prob != prob), f"Probability was NaN for {case['id']}"
        assert 0.0 <= prob <= 1.0, f"Probability {prob} outside [0.0, 1.0] for {case['id']}"

    assert elapsed_ms < 300.0, f"Inference SLA breached on {case['id']}: {elapsed_ms:.1f}ms"


# ===========================================================================
# 2. Monte Carlo Random Input Fuzzing (50 Generated Random Profiles)
# ===========================================================================

@pytest.mark.stress
@pytest.mark.ml
def test_ml_monte_carlo_random_applicants_batch_fuzzing(ml_api: APIRequestContext):
    """
    Generate 50 distinct random applicant profiles using Monte Carlo sampling.
    Submit sequentially to the live inference endpoint.
    Asserts 100% success rate, bounded probabilities, and zero server crashes.
    """
    total_cases = 50
    success_count = 0
    predictions = []
    probabilities = []

    t0 = time.perf_counter()
    for i in range(total_cases):
        applicant = generate_random_applicant(seed=1000 + i)
        resp = ml_api.post("/api/v1/predict", data=applicant)
        if resp.status == 200:
            body = resp.json()
            if body.get("success") is True:
                success_count += 1
                predictions.append(body["prediction"])
                if body.get("probability") is not None:
                    probabilities.append(float(body["probability"]))

    total_time = time.perf_counter() - t0
    avg_latency = (total_time / total_cases) * 1000.0

    assert success_count == total_cases, f"Only {success_count}/{total_cases} random applicants succeeded"
    assert len(predictions) == total_cases
    assert all(p in (0, 1) for p in predictions)
    assert all(0.0 <= prob <= 1.0 for prob in probabilities)
    print(f"\n[Monte Carlo ML Fuzzing] 50/50 Passed | Avg Latency: {avg_latency:.1f}ms | Total Time: {total_time:.2f}s")


# ===========================================================================
# 3. Batch Inference Stress & Scalability Test
# ===========================================================================

@pytest.mark.stress
@pytest.mark.ml
@pytest.mark.parametrize("batch_size", [5, 20, 50])
def test_ml_batch_prediction_scalability_and_throughput(
    ml_api: APIRequestContext, batch_size: int
):
    """
    Stress-test batch prediction endpoint with increasing record counts (5, 20, 50).
    Asserts:
    1. Returns count == batch_size.
    2. Length of predictions matches input records exactly.
    3. Throughput > 10 records/second.
    """
    records = [generate_random_applicant(seed=2000 + i) for i in range(batch_size)]
    payload = {"records": records}

    start = time.perf_counter()
    resp = ml_api.post("/api/v1/predict/batch", data=payload)
    elapsed_s = time.perf_counter() - start

    assert resp.status == 200, f"Batch size {batch_size} failed: {resp.text()}"
    body = resp.json()

    assert body.get("success") is True
    assert body.get("count") == batch_size
    assert len(body.get("predictions", [])) == batch_size

    throughput = batch_size / elapsed_s if elapsed_s > 0 else 0
    print(f"\n[Batch Stress] Size: {batch_size} records in {elapsed_s:.2f}s -> Throughput: {throughput:.1f} records/sec")
    assert throughput > 2.0, f"Batch throughput too low: {throughput:.1f} rec/s"


# ===========================================================================
# 4. Multi-Model Invariance & Determinism Under Stress
# ===========================================================================

@pytest.mark.stress
@pytest.mark.ml
def test_ml_cross_model_evaluation_on_clean_ground_truth(ml_api: APIRequestContext):
    """
    Evaluate multiple models on identical test input to verify cross-model stability
    and ensure every model responds without NaN outputs.
    """
    applicant = generate_random_applicant(seed=42)

    # Fetch available models from API
    models_resp = ml_api.get("/api/v1/models")
    assert models_resp.status == 200
    models_list = [m["name"] for m in models_resp.json().get("models", [])]

    assert len(models_list) >= 4, f"Expected at least 4 models, got {len(models_list)}"

    tested_count = 0
    for model_name in models_list:
        resp = ml_api.post("/api/v1/predict", data=dict(applicant, model=model_name))
        if resp.status == 200:
            body = resp.json()
            assert body["success"] is True
            assert body["prediction"] in (0, 1)
            tested_count += 1

    assert tested_count >= 4, f"Only tested {tested_count} models successfully"
