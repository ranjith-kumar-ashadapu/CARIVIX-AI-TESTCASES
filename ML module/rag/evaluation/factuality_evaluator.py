"""Groundedness / factuality evaluation for generated answers."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("CARIVIX_AI")


class FactualityEvaluator:
    """Identify supported and unsupported claims based on retrieved context."""

    def __init__(self) -> None:
        self.logger = logger

    def evaluate(self, answer: str, context: List[str], query: str) -> Dict[str, Any]:
        if not answer or not answer.strip():
            return {
                "query": query,
                "supported_claims": [],
                "unsupported_claims": ["No answer generated."],
                "missing_context": True,
                "potential_hallucinations": ["No answer available for factual evaluation."],
            }

        joined_context = "\n".join(context).lower()
        supported: List[str] = []
        unsupported: List[str] = []

        for sentence in [part.strip() for part in answer.split(".") if part.strip()]:
            if not sentence:
                continue
            if any(term in joined_context for term in sentence.lower().split()[:5]):
                supported.append(sentence)
            else:
                unsupported.append(sentence)

        missing_context = bool(not context or not joined_context.strip())
        returns = {
            "query": query,
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "missing_context": missing_context,
            "potential_hallucinations": unsupported if unsupported else [],
        }
        return returns
