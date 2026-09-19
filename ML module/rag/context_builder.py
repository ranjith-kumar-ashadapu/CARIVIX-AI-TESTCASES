"""Context builder for grounded RAG responses."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger("CARIVIX_AI")


class ContextBuilder:
    """Build and normalize retrieved context before sending it to an LLM."""

    def __init__(self, max_context_chars: Optional[int] = None) -> None:
        self.max_context_chars = max_context_chars

    def build(self, retrieved_chunks: Iterable[Dict[str, Any]], max_context_chars: Optional[int] = None) -> str:
        """Combine retrieved chunks into a single context string."""
        chunks = list(retrieved_chunks or [])
        if not chunks:
            return "No relevant context available."

        limit = max_context_chars if max_context_chars is not None else self.max_context_chars
        context_parts: List[str] = []
        total_chars = 0
        seen_texts = set()

        for chunk in chunks:
            text = chunk.get("text") or chunk.get("document")
            if hasattr(text, "page_content"):
                text = text.page_content
            text = str(text or "").strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)

            source = chunk.get("source") or "unknown"
            chunk_id = chunk.get("chunk_id")
            chunk_index = chunk.get("chunk_index")
            document_id = chunk.get("document_id")
            page = chunk.get("page")

            metadata_bits = [f"Source: {source}"]
            if document_id:
                metadata_bits.append(f"Document ID: {document_id}")
            if chunk_id is not None:
                metadata_bits.append(f"Chunk ID: {chunk_id}")
            if chunk_index is not None:
                metadata_bits.append(f"Chunk Index: {chunk_index}")
            if page is not None:
                metadata_bits.append(f"Page: {page}")

            header = " | ".join(metadata_bits)
            formatted = f"{header}\n{text}"
            if limit is not None and total_chars + len(formatted) > limit:
                remaining = max(limit - total_chars, 0)
                if remaining > 80:
                    context_parts.append(formatted[:remaining].rstrip() + "...")
                break
            context_parts.append(formatted)
            total_chars += len(formatted)

        if not context_parts:
            return "No relevant context available."

        return "\n\n---\n\n".join(context_parts)
