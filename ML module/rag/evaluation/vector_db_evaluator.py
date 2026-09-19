"""
Vector Database Integration Evaluator
=======================================

Validates the FAISS vector database integration of the existing CARIVIX AI
RAG pipeline without modifying it.

Responsibilities:
    - Load the persisted FAISS index (index.faiss + index.pkl).
    - Verify the index loads successfully and report vector counts.
    - Confirm embeddings are stored with the expected dimension.
    - Run similarity search using multiple sample queries.
    - Display document IDs, similarity scores, and retrieved text.
    - Validate retrieved-document relevance (lexical overlap proxy).
    - Detect duplicate retrieval results.
    - Produce a structured integration status summary.

Usage::

    evaluator = VectorDBEvaluator(
        vector_store_dir="data/vector_store/",
        embedding_model="all-MiniLM-L6-v2",
    )
    summary = evaluator.run_evaluation()
    evaluator.print_summary()
"""

import os
import logging
from typing import Any, Dict, List, Optional

from rag.embeddings import EmbeddingGenerator
from rag.vector_store import VectorStore
from rag.retriever import Retriever

from rag.evaluation.utils import (
    Timer,
    format_elapsed,
    lexical_overlap,
    render_table,
)

logger = logging.getLogger("CARIVIX_AI")

class VectorDBEvaluator:
    """
    Evaluates the integrity and retrieval quality of the FAISS vector store.

    Attributes:
        vector_store_dir: Directory containing the FAISS index.
        embedding_model: Sentence Transformer model name.
        embedding_generator: Lazily-loaded EmbeddingGenerator.
        vector_store: Lazily-loaded VectorStore.
        retriever: Retriever combining the embedding generator + vector store.
        sample_queries: Queries used to exercise similarity search.
    """

    def __init__(
        self,
        vector_store_dir: str = "data/vector_store/",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_device: Optional[str] = None,
        sample_queries: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> None:
        """
        Initialize the VectorDBEvaluator.

        Args:
            vector_store_dir: Path to the persisted FAISS index directory.
            embedding_model: Embedding model name (default all-MiniLM-L6-v2).
            embedding_device: Device for embeddings ('cpu' or 'cuda').
            sample_queries: Queries to run for similarity search validation.
            top_k: Number of results to retrieve per query.
        """
        self.vector_store_dir = vector_store_dir
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.top_k = top_k

        self.sample_queries = sample_queries or [
            "What is CARIVIX AI?",
            "What machine learning algorithms are supported?",
            "How does the platform handle data preprocessing?",
            "What is economic analysis?",
            "Tell me about market prediction.",
        ]

        # Lazy-loaded components
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self.vector_store: Optional[VectorStore] = None
        self.retriever: Optional[Retriever] = None

        # Evaluation results
        self._summary: Dict[str, Any] = {}
        self._search_results: List[Dict[str, Any]] = []

        logger.info(
            "VectorDBEvaluator initialized. Store: %s, Model: %s, Top-k: %d",
            vector_store_dir,
            embedding_model,
            top_k,
        )

    # ------------------------------------------------------------------
    # Component Setup
    # ------------------------------------------------------------------

    def _ensure_components(self) -> None:
        """Lazily build the embedding generator, vector store, and retriever."""
        if self.embedding_generator is None:
            self.embedding_generator = EmbeddingGenerator(
                model_name=self.embedding_model,
                device=self.embedding_device,
            )
        if self.vector_store is None:
            self.vector_store = VectorStore(
                index_directory=self.vector_store_dir,
            )
        if self.retriever is None:
            self.retriever = Retriever(
                embedding_generator=self.embedding_generator,
                vector_store=self.vector_store,
            )

    # ------------------------------------------------------------------
    # FAISS Index Validation
    # ------------------------------------------------------------------

    def _validate_index(self) -> Dict[str, Any]:
        """
        Load the FAISS index and validate its integrity.

        Returns:
            A dictionary describing index status, vector count, dimension,
            and whether the embedding dimension matches expectations.
        """
        self._ensure_components()

        result = {
            "index_loaded": False,
            "status": "FAIL",
            "vector_count": 0,
            "dimension": None,
            "expected_dimension": self.embedding_generator.EXPECTED_DIMENSION,
            "document_count": 0,
            "index_path": None,
            "metadata_path": None,
            "errors": [],
        }

        index_path = os.path.join(self.vector_store_dir, "index.faiss")
        metadata_path = os.path.join(self.vector_store_dir, "index.pkl")

        result["index_path"] = index_path
        result["metadata_path"] = metadata_path

        # --- Check files exist ---
        if not os.path.exists(index_path):
            result["errors"].append(f"FAISS index file not found: {index_path}")
        if not os.path.exists(metadata_path):
            result["errors"].append(f"Metadata file not found: {metadata_path}")

        if result["errors"]:
            logger.error("Index validation failed: %s", result["errors"])
            return result

        # --- Load index ---
        try:
            with Timer("FAISS index load") as t:
                self.vector_store.load_index()
            result["index_load_time"] = round(t["elapsed"], 6)
        except Exception as exc:
            result["errors"].append(f"Failed to load index: {exc}")
            logger.error("Failed to load index: %s", exc)
            return result

        result["index_loaded"] = True
        result["vector_count"] = self.vector_store.vector_count
        result["dimension"] = self.vector_store.dimension
        result["document_count"] = len(self.vector_store.documents)

        # --- Validate dimension ---
        expected = self.embedding_generator.EXPECTED_DIMENSION
        if result["dimension"] == expected:
            result["dimension_match"] = True
        else:
            result["dimension_match"] = False
            result["errors"].append(
                f"Dimension mismatch: index={result['dimension']}, "
                f"expected={expected}"
            )

        # --- Validate vector/document alignment ---
        if result["vector_count"] != result["document_count"]:
            result["errors"].append(
                f"Vector/document mismatch: {result['vector_count']} vectors "
                f"vs {result['document_count']} documents"
            )

        # --- Determine status ---
        if result["vector_count"] > 0 and result["dimension_match"] and not result["errors"]:
            result["status"] = "PASS"
        else:
            result["status"] = "WARN"

        logger.info(
            "Index validation: status=%s, vectors=%d, dimension=%s",
            result["status"],
            result["vector_count"],
            result["dimension"],
        )
        return result

    # ------------------------------------------------------------------
    # Similarity Search
    # ------------------------------------------------------------------

    def _run_similarity_searches(self) -> List[Dict[str, Any]]:
        """
        Run similarity search for each sample query.

        Returns:
            A list of per-query result records including scores, chunk IDs,
            sources, predicted relevance, and duplicate detection.
        """
        self._ensure_components()
        records: List[Dict[str, Any]] = []

        for query in self.sample_queries:
            with Timer(f"Search: {query[:40]}") as t:
                results = self.retriever.retrieve(query, k=self.top_k)

            record = {
                "query": query,
                "retrieval_time": round(t["elapsed"], 6),
                "num_results": len(results),
                "results": results,
                "avg_score": 0.0,
                "lexical_relevance": 0.0,
                "duplicates": 0,
                "duplicate_chunk_ids": [],
            }

            if results:
                scores = [r["score"] for r in results]
                record["avg_score"] = round(sum(scores) / len(scores), 6)
                record["lexical_relevance"] = round(
                    sum(
                        lexical_overlap(query, r["document"].page_content)
                        for r in results
                    ) / len(results),
                    6,
                )

                # Detect duplicate chunks within the result set.
                # NOTE: chunk_id is reset per source document (see splitter.py),
                # so we must key duplicates on (source, chunk_id) to avoid
                # false positives across different documents.
                seen: Dict[Any, int] = {}
                duplicate_ids: List[Any] = []
                for r in results:
                    identity = (r["source"], r["chunk_id"])
                    if identity in seen:
                        duplicate_ids.append(identity)
                    else:
                        seen[identity] = 1
                record["duplicates"] = len(duplicate_ids)
                record["duplicate_chunk_ids"] = duplicate_ids

            records.append(record)
            logger.info(
                "Query '%s' -> %d results, avg_score=%.4f, dup=%d",
                query[:50],
                len(results),
                record["avg_score"],
                record["duplicates"],
            )

        return records

    # ------------------------------------------------------------------
    # Full Evaluation
    # ------------------------------------------------------------------

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the complete vector database integration evaluation.

        Returns:
            A structured summary dictionary containing:
                - index_valid:  FAISS index validation results
                - searches:     Per-query similarity search results
                - duplicates:   Aggregate duplicate detection statistics
                - relevance:    Aggregate lexical relevance statistics
                - overall_status: PASS / WARN / FAIL
        """
        logger.info("=" * 70)
        logger.info("  VECTOR DATABASE INTEGRATION EVALUATION")
        logger.info("=" * 70)

        index_valid = self._validate_index()
        if not index_valid.get("index_loaded"):
            self._summary = {
                "index_valid": index_valid,
                "searches": [],
                "overall_status": "FAIL",
                "error": "Index could not be loaded.",
            }
            return self._summary

        searches = self._run_similarity_searches()

        # Aggregate statistics
        all_avg_scores = [s["avg_score"] for s in searches if s["num_results"] > 0]
        all_lexical = [s["lexical_relevance"] for s in searches if s["num_results"] > 0]
        total_duplicates = sum(s["duplicates"] for s in searches)
        total_queries = len(searches)

        # Determine overall status
        if all_avg_scores and min(all_avg_scores) > 0.3 and total_duplicates == 0:
            overall = "PASS"
        elif all_avg_scores and min(all_avg_scores) > 0.3:
            overall = "WARN"
        else:
            overall = "WARN"

        if index_valid.get("status") == "FAIL":
            overall = "FAIL"

        self._summary = {
            "index_valid": index_valid,
            "searches": searches,
            "total_queries": total_queries,
            "total_duplicates": total_duplicates,
            "avg_score_overall": round(
                sum(all_avg_scores) / len(all_avg_scores), 6
            ) if all_avg_scores else 0.0,
            "avg_lexical_overall": round(
                sum(all_lexical) / len(all_lexical), 6
            ) if all_lexical else 0.0,
            "overall_status": overall,
        }
        self._search_results = searches

        logger.info(
            "Vector DB evaluation complete. Status: %s, Queries: %d, "
            "Duplicates: %d",
            overall,
            total_queries,
            total_duplicates,
        )
        return self._summary

    # ------------------------------------------------------------------
    # Reporting Helpers
    # ------------------------------------------------------------------

    @property
    def summary(self) -> Dict[str, Any]:
        """Return the stored evaluation summary."""
        return self._summary

    def get_index_summary_table(self) -> str:
        """Render the index validation summary as a console table."""
        iv = self._summary.get("index_valid", {})
        rows = [
            ["FAISS Index Loaded", "Yes" if iv.get("index_loaded") else "No"],
            ["Index Status", iv.get("status", "N/A")],
            ["Vector Count", iv.get("vector_count", 0)],
            ["Document Count", iv.get("document_count", 0)],
            ["Embedding Dimension", iv.get("dimension", "N/A")],
            ["Expected Dimension", iv.get("expected_dimension", "N/A")],
            ["Dimension Match", "Yes" if iv.get("dimension_match") else "No"],
            ["Index Load Time", format_elapsed(iv.get("index_load_time", 0))],
            ["Index Path", iv.get("index_path", "N/A")],
            ["Metadata Path", iv.get("metadata_path", "N/A")],
        ]
        return render_table(["Property", "Value"], rows, title="FAISS INDEX VALIDATION")

    def get_similarity_search_table(self) -> str:
        """Render a concise per-query similarity search summary table."""
        rows = []
        for s in self._summary.get("searches", []):
            rows.append([
                s["query"][:45],
                s["num_results"],
                f"{s['avg_score']:.4f}",
                f"{s['lexical_relevance']:.4f}",
                s["duplicates"],
                format_elapsed(s["retrieval_time"]),
            ])
        return render_table(
            ["Query", "Results", "Avg Score", "Lexical Rel.", "Duplicates", "Retrieval"],
            rows,
            title="SIMILARITY SEARCH RESULTS",
        )

    def get_detailed_results_table(self) -> str:
        """
        Render a detailed table of every retrieved document per query.

        Includes document ID (chunk_id), similarity score, source, and a
        short text preview.
        """
        rows = []
        for s in self._summary.get("searches", []):
            for r in s["results"]:
                text = r["document"].page_content.replace("\n", " ")[:60]
                rows.append([
                    s["query"][:30],
                    r["rank"],
                    r["chunk_id"],
                    r["source"],
                    f"{r['score']:.4f}",
                    text + "...",
                ])
        return render_table(
            ["Query", "Rank", "Chunk ID", "Source", "Score", "Text Preview"],
            rows,
            title="RETRIEVED DOCUMENTS (DETAILED)",
            max_width=160,
        )

    def get_duplicate_report_table(self) -> str:
        """Render a table of duplicate detection results."""
        rows = []
        for s in self._summary.get("searches", []):
            if s["duplicates"] > 0:
                rows.append([
                    s["query"][:45],
                    s["duplicates"],
                    ", ".join(str(c) for c in s["duplicate_chunk_ids"]),
                ])
        if not rows:
            rows.append(["No duplicates detected across all queries", "-", "-"])
        return render_table(
            ["Query", "Duplicates", "Duplicate Chunk IDs"],
            rows,
            title="DUPLICATE DETECTION",
        )

    def print_summary(self) -> None:
        """Print the full vector DB evaluation summary to the console."""
        print("\n" + self.get_index_summary_table())
        print("\n" + self.get_similarity_search_table())
        print("\n" + self.get_duplicate_report_table())
        print("\n" + self.get_detailed_results_table())

        overall = self._summary.get("overall_status", "N/A")
        print(f"\n{'=' * 70}")
        print(f"  OVERALL VECTOR DB STATUS: {overall}")
        print(f"{'=' * 70}")

