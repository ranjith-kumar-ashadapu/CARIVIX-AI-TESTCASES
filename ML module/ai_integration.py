"""
AI integration routes: hooks nlp_module + RAG pipeline + ML service together.

Provides register_ai_routes(app) which adds POST /api/v1/ai/query to the FastAPI app.
The endpoint behavior:
  - Runs nlp_module.analyze(query)
  - If route == 'rag': uses RAGPipeline to retrieve context and generate an LLM response
  - If route == 'ml': instructs user to call /api/v1/predict (model inputs are domain-specific)
  - If route == 'combined': attempts to retrieve RAG context, calls ML only if structured input

This module is intentionally conservative about calling the ModelService.predict API
because the ModelService expects structured tabular input. For production, adapt the
input-mapping logic here to convert natural-language queries into the model's feature dict.
"""
from typing import Any, Dict, Optional
import logging
import time
import re
import json
from collections import Counter
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from rag.pipeline import RAGPipeline
from nlp_module import analyze as nlp_analyze

# Load mapping examples to derive sensible defaults for categorical features
_MAPPING_PATH = "mapping_examples.json"
_mapping_examples = []
_categorical_defaults: Dict[str, Any] = {}
try:
    with open(_MAPPING_PATH, "r", encoding="utf-8") as f:
        _mapping_examples = json.load(f)
    # find categorical fields by inspecting example values that are strings
    cat_counters: Dict[str, Counter] = {}
    for ex in _mapping_examples:
        feats = ex.get("features", {})
        for k, v in feats.items():
            if isinstance(v, str):
                if k not in cat_counters:
                    cat_counters[k] = Counter()
                cat_counters[k][v] += 1
    for k, cnt in cat_counters.items():
        _categorical_defaults[k] = cnt.most_common(1)[0][0]
except Exception:
    _mapping_examples = []
    _categorical_defaults = {}

logger = logging.getLogger("CARIVIX_AI.ai_integration")


class AIQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: Optional[int] = Field(None, description="Number of RAG chunks to retrieve")
    max_tokens: Optional[int] = Field(512, description="LLM max tokens")
    temperature: Optional[float] = Field(0.0, description="LLM temperature")
    auto_fill_missing: Optional[bool] = Field(False, description="If true, automatically fill missing features with defaults when running ML prediction")


class AIQueryResponse(BaseModel):
    success: bool = True
    nlp: Dict[str, Any]
    selected_route: str
    rag_context: Optional[Dict[str, Any]] = None
    ml_prediction: Optional[Dict[str, Any]] = None
    final_response: Optional[str] = None


import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def _get_rag_pipeline(app: FastAPI) -> RAGPipeline:
    # Lazy-init pipeline and stash in app.state
    if not hasattr(app.state, "rag_pipeline") or app.state.rag_pipeline is None:
        logger.info("Initializing RAGPipeline (lazy)...")
        docs_dir = os.path.join(PROJECT_ROOT, "data", "documents")
        vec_dir = os.path.join(PROJECT_ROOT, "data", "vector_store")
        pipeline = RAGPipeline(
            documents_dir=docs_dir,
            vector_store_dir=vec_dir,
            llm_backend="ollama",
            llm_model=None,
        )
        app.state.rag_pipeline = pipeline
    return app.state.rag_pipeline


def register_ai_routes(app: FastAPI) -> None:
    """Register AI integration endpoints on the given FastAPI app."""

    @app.post("/api/v1/ai/query", response_model=AIQueryResponse, summary="Run integrated NLP → RAG → (ML) query")
    async def ai_query(payload: AIQueryRequest):
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty query")

        # Step 1: NLP analysis
        nlp_out = nlp_analyze(query)
        route = nlp_out.get("route", "rag")

        response_payload: Dict[str, Any] = {
            "success": True,
            "nlp": nlp_out,
            "selected_route": route,
            "rag_context": None,
            "ml_prediction": None,
            "final_response": None,
        }

        # Step 2: Route handling
        if route == "rag":
            try:
                pipeline = _get_rag_pipeline(app)
                total_start = time.time()
                result = pipeline.query(
                    question=query,
                    k=payload.k,
                    max_tokens=payload.max_tokens or 512,
                    temperature=payload.temperature or 0.0,
                    verbose=False,
                )
                total_time = time.time() - total_start
                response_payload["rag_context"] = {
                    "retrieved_chunks": result.get("retrieved_chunks", []),
                    "retrieval_time": result.get("retrieval_time"),
                    "generation_time": result.get("generation_time"),
                    "total_time": result.get("total_time", total_time),
                }
                response_payload["final_response"] = result.get("response")
                return response_payload
            except RuntimeError as exc:
                logger.exception("RAG pipeline failed: %s", exc)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

        elif route == "ml":
            # Attempt a conservative mapping from natural-language to model features and run prediction
            model_service = getattr(app.state, "model_service", None)
            if model_service is None:
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model service unavailable")

            def _extract_features_from_text(q: str, expected_features: Optional[list]) -> Dict[str, Any]:
                """Try to extract simple numeric and categorical features from free text.

                This is a heuristic bootstrap: it recognizes numbers for numeric features
                and a small set of keywords for categorical fields. If extraction is
                insufficient (too few features), it returns an empty dict.
                """
                features: Dict[str, Any] = {}
                q_low = q.lower()

                # Common numeric patterns
                nums = [int(n.replace(',', '')) for n in re.findall(r"\b\d{1,3}(?:,\d{3})*\b", q)]
                # Credit-score-like 3-digit numbers
                three_digit = re.findall(r"\b\d{3}\b", q)

                for feat in expected_features or []:
                    f = feat.lower()
                    if "age" in f:
                        m = re.search(r"age\s*(?:is|:)?\s*(\d{1,3})", q_low)
                        if m:
                            features[feat] = float(m.group(1))
                        elif nums:
                            features[feat] = float(nums[0])
                    elif any(x in f for x in ["income", "salary", "pay"]):
                        m = re.search(r"(?:income|salary|pay)\s*(?:is|:)?\s*\$?([0-9,]+)", q_low)
                        if m:
                            features[feat] = float(m.group(1).replace(',', ''))
                        elif nums:
                            features[feat] = float(nums[0])
                    elif "credit" in f or "score" in f:
                        if three_digit:
                            features[feat] = float(three_digit[0])
                    elif any(x in f for x in ["loan", "amount"]):
                        m = re.search(r"(?:loan|amount)\s*(?:is|:)?\s*\$?([0-9,]+)", q_low)
                        if m:
                            features[feat] = float(m.group(1).replace(',', ''))
                        elif nums:
                            features[feat] = float(nums[0])
                    elif any(x in f for x in ["years", "employ"]):
                        m = re.search(r"(\d+)\s*(?:years|yrs)\s*(?:employed)?", q_low)
                        if m:
                            features[feat] = float(m.group(1))
                        elif nums:
                            features[feat] = float(nums[0])
                    else:
                        # Small categorical heuristics
                        if "bachelor" in q_low or "ba" in q_low:
                            features[feat] = "bachelor"
                        elif "master" in q_low or "ms" in q_low:
                            features[feat] = "master"
                        elif "phd" in q_low:
                            features[feat] = "phd"
                        elif "employed" in q_low:
                            features[feat] = "employed"
                        elif "self-employed" in q_low or "self employed" in q_low:
                            features[feat] = "self-employed"
                        elif "unemployed" in q_low:
                            features[feat] = "unemployed"
                        elif "married" in q_low:
                            features[feat] = "married"
                        elif "single" in q_low:
                            features[feat] = "single"
                return features

            # Determine raw/system-level required input fields from ModelService if available
            required_fields = [
                "age",
                "income",
                "credit_score",
                "loan_amount",
                "years_employed",
                "education",
                "employment_status",
                "marital_status",
                "housing_type",
                "application_date",
            ]
            try:
                from src.model_service import _REQUIRED_SYSTEM_FIELDS as _REQ
            except Exception:
                _REQ = None
            if _REQ and isinstance(_REQ, list):
                required_fields = list(_REQ)

            # Extract only the base/system fields from the query
            extracted_base = _extract_features_from_text(query, required_fields)

            missing_base = [f for f in required_fields if f not in extracted_base or extracted_base.get(f) in (None, "")]
            if missing_base:
                if payload.auto_fill_missing:
                    # Auto-fill missing base fields: use categorical defaults or safe numeric defaults
                    for mb in missing_base:
                        if mb == "application_date":
                            extracted_base[mb] = datetime.utcnow().date().isoformat()
                        elif mb in _categorical_defaults:
                            extracted_base[mb] = _categorical_defaults.get(mb)
                        else:
                            # numeric default
                            extracted_base[mb] = 0 if any(tok in mb for tok in ["age", "income", "score", "loan", "years"]) else "unknown"
                else:
                    response_payload["ml_prediction"] = {
                        "message": "Automatic extraction incomplete.",
                        "extracted": extracted_base,
                        "missing_fields": missing_base,
                    }
                    response_payload["final_response"] = (
                        "ML routing selected, but automatic feature extraction did not find all required fields. "
                        "Please call /api/v1/predict with the full set of model features or enable auto_fill_missing."
                    )
                    return response_payload

            # Now prepared raw input should contain all required base fields; attempt prediction
            try:
                pred = model_service.predict(extracted_base, model_name=None)
                response_payload["ml_prediction"] = pred
                response_payload["final_response"] = f"ML prediction: {pred.get('prediction')}"
                return response_payload
            except Exception as exc:
                logger.exception("Automatic ML prediction failed: %s", exc)
                response_payload["ml_prediction"] = {"error": str(exc)}
                response_payload["final_response"] = "ML prediction failed. See ml_prediction.error for details."
                return response_payload

        elif route == "combined":
            # Attempt to run retrieval and include ML placeholder; full ML integration requires mapping
            pipeline = _get_rag_pipeline(app)
            try:
                retr_res = pipeline.retriever.retrieve(query=query, k=payload.k or pipeline.retrieval_k)
                prompt = pipeline.prompt_builder.build_prompt(query=query, retrieved_chunks=retr_res)

                # Try to include ML prediction when structured input is available; otherwise return combined context
                model_service = getattr(app.state, "model_service", None)
                ml_pred = None
                if model_service and False:
                    # Placeholder: domain-specific mapping would go here
                    pass

                # Generate response with the prompt and any appended ML prediction info
                gen = pipeline.generator
                if gen.is_available():
                    final = gen.generate(prompt=prompt, max_tokens=payload.max_tokens or 512, temperature=payload.temperature or 0.0)
                else:
                    final = "[LLM not available. Retrieved context returned.]"

                response_payload["rag_context"] = {"retrieved_chunks": retr_res}
                response_payload["ml_prediction"] = ml_pred
                response_payload["final_response"] = final
                return response_payload
            except Exception as exc:
                logger.exception("Combined route failed: %s", exc)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported route: {route}")

    # Endpoint registration complete
    logger.info("AI integration routes registered.")
