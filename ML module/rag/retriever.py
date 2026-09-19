"""
Retriever Module for CARIVIX AI RAG Pipeline
==============================================

Performs similarity search over the vector store to retrieve the
most relevant document chunks for a given user query.

Flow:
    User Query
        ↓
    Generate Query Embedding
        ↓
    Search FAISS Index (Top-K)
        ↓
    Return Ranked Chunks with Scores

Usage:
    retriever = Retriever(embedding_generator, vector_store)
    results = retriever.retrieve("What is CARIVIX AI?", k=5)
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from rag.embeddings import EmbeddingGenerator
from rag.vector_store import VectorStore

logger = logging.getLogger("CARIVIX_AI")

class Retriever:
    """
    Retrieves the most relevant document chunks for a given query.

    Combines an EmbeddingGenerator (to embed the query) with a
    VectorStore (to perform similarity search) and returns ranked
    chunks with metadata.

    Attributes:
        embedding_generator: EmbeddingGenerator instance.
        vector_store: VectorStore instance with a loaded index.
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
    ) -> None:
        """
        Initialize the Retriever.

        Args:
            embedding_generator: An initialized EmbeddingGenerator.
            vector_store: An initialized VectorStore (index should be
                created or loaded before calling retrieve).
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store

        logger.info(
            "Retriever initialized. Embedding model: %s, "
            "Vector store: %s",
            embedding_generator.model_name,
            vector_store.index_directory,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query: User query string.
            k: Number of chunks to retrieve.
            score_threshold: Minimum similarity score threshold.
                Results below this threshold are excluded.

        Returns:
            List of dictionaries, each containing:
                - 'rank':       Rank position (1-based)
                - 'score':      Similarity score (0-1, higher = better)
                - 'distance':   Raw FAISS distance
                - 'document':   The Document object
                - 'chunk_id':   Chunk identifier
                - 'source':     Source document name
                - 'page':       Page number (if applicable)
                - 'text':       Shortcut to document.page_content
        """
        if not query or not query.strip():
            logger.warning("Empty query received.")
            return []

        if self.vector_store is None or self.vector_store.index is None:
            logger.warning("Retriever cannot run because the FAISS index is not loaded.")
            return []

        logger.info("Retrieving top-%d chunks for query: '%s'", k, query[:100])

        start_time = time.time()
        query_vector = self.embedding_generator.generate_query_embedding(query)
        embed_time = time.time() - start_time
        logger.debug("Query embedding generated in %.4f seconds.", embed_time)

        search_start = time.time()
        try:
            results = self.vector_store.similarity_search(query_vector, k=k)
        except ValueError as exc:
            logger.warning("Retrieval failed: %s", exc)
            return []
        search_time = time.time() - search_start
        logger.debug("Vector search completed in %.4f seconds.", search_time)

        # Step 3: Apply score threshold
        if score_threshold is not None:
            results = [
                r for r in results
                if r["score"] >= score_threshold
            ]
            logger.debug(
                "After score threshold filter: %d results.",
                len(results),
            )

        for result in results:
            document = result.get("document")
            if document is not None:
                result["text"] = document.page_content
            if "metadata" not in result:
                result["metadata"] = dict(getattr(document, "metadata", {}) or {})
            result.setdefault("source", result["metadata"].get("source", "unknown"))
            result.setdefault("document_id", result["metadata"].get("document_id"))
            result.setdefault("file_name", result["metadata"].get("file_name") or result.get("source"))
            result.setdefault("document_type", result["metadata"].get("document_type", "text"))
            result.setdefault("chunk_index", result["metadata"].get("chunk_index"))

        logger.info(
            "Retrieval complete. %d results in %.4f seconds "
            "(embed: %.4fs, search: %.4fs).",
            len(results),
            embed_time + search_time,
            embed_time,
            search_time,
        )

        return results

    def retrieve_with_scores(
        self,
        query: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve results with explicit similarity scores highlighted.

        This is a convenience wrapper around retrieve() that ensures
        scores are prominently displayed.

        Args:
            query: User query string.
            k: Number of chunks to retrieve.

        Returns:
            Same as retrieve(), with scores between 0 and 1.
        """
        return self.retrieve(query, k=k)

    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve results for multiple queries at once.

        Args:
            queries: List of query strings.
            k: Number of chunks to retrieve per query.

        Returns:
            List of result lists, one per query.
        """
        logger.info(
            "Batch retrieval for %d queries.", len(queries)
        )
        all_results = []
        for idx, query in enumerate(queries):
            logger.debug(
                "Batch query %d/%d: '%s'",
                idx + 1, len(queries), query[:50],
            )
            results = self.retrieve(query, k=k)
            all_results.append(results)
        return all_results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def format_results(
        self,
        results: List[Dict[str, Any]],
        include_text: bool = True,
    ) -> str:
        """
        Format retrieval results as a human-readable string.

        Args:
            results: List of result dictionaries from retrieve().
            include_text: Whether to include the chunk text content.

        Returns:
            Formatted string for display.
        """
        if not results:
            return "No relevant documents found."

        lines = []
        lines.append("=" * 70)
        lines.append("  RETRIEVAL RESULTS")
        lines.append("=" * 70)

        for r in results:
            lines.append(
                f"\n  Rank {r['rank']} | "
                f"Score: {r['score']:.4f} | "
                f"Source: {r['source']}"
            )
            if r.get("page"):
                lines.append(f"  Page: {r['page']}")
            lines.append(f"  Chunk: {r['chunk_id']}")
            if include_text:
                text = r['text'][:200]
                lines.append(f"  Text: {text}...")
            lines.append("-" * 70)

        return "\n".join(lines)

