"""Retrieval relevance evaluation for RAG queries."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("CARIVIX_AI")


class RelevanceEvaluator:
    """Evaluate whether retrieved chunks are relevant to the user's question."""

    def __init__(self) -> None:
        self.logger = logger

    def evaluate(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not query or not query.strip():
            return {"query": query, "retrieved_documents": [], "relevance_score": 0.0, "relevance_status": "empty_query"}

        if not retrieved_chunks:
            return {"query": query, "retrieved_documents": [], "relevance_score": 0.0, "relevance_status": "no_context"}

        scores: List[float] = []
        for chunk in retrieved_chunks:
            text = chunk.get("text") or chunk.get("document")
            if hasattr(text, "page_content"):
                text = text.page_content
            text = str(text or "").lower()
            query_terms = set(tok for tok in query.lower().split() if tok.isalpha())
            if not query_terms:
                scores.append(0.0)
                continue
            overlap = sum(1 for term in query_terms if term in text)
            score = overlap / max(len(query_terms), 1)
            scores.append(score)

        overall = sum(scores) / max(len(scores), 1)
        status = "relevant" if overall >= 0.2 else "not_relevant"
        return {
            "query": query,
            "retrieved_documents": [chunk.get("source") or "unknown" for chunk in retrieved_chunks],
            "relevance_score": round(float(overall), 4),
            "relevance_status": status,
        }
