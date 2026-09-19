"""
Retrieval Performance Optimizer
================================

Benchmarks the existing CARIVIX AI RAG pipeline across a grid of
configuration values without modifying the production pipeline or its
persisted index.

Each configuration is evaluated using an in-memory FAISS index built from
the same source documents, so the production `data/vector_store/` is never
overwritten.

Sweep grid:
    - Chunk Size:   300, 500, 800
    - Chunk Overlap: 30, 50, 100
    - Top-k:         3, 5, 10

Measured metrics per configuration:
    - Embedding generation time
    - FAISS indexing time
    - Retrieval latency (per query)
    - Total query response time (retrieval + query embedding)
    - Average similarity score of retrieved documents
    - Lexical relevance (retrieval quality proxy)

Usage::

    evaluator = PerformanceEvaluator(
        documents_dir="data/documents/",
        vector_store_dir="data/vector_store/",
        embedding_model="all-MiniLM-L6-v2",
    )
    results = evaluator.run_sweep()
    evaluator.print_results()
    best = evaluator.get_best_config()
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from rag.embeddings import EmbeddingGenerator
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.loader import DocumentLoader
from rag.splitter import TextPreprocessor, DocumentSplitter

from rag.evaluation.utils import (
    Timer,
    format_elapsed,
    lexical_overlap,
    render_table,
)

logger = logging.getLogger("CARIVIX_AI")

class PerformanceEvaluator:
    """
    Runs a configuration sweep over the RAG pipeline and picks the best config.

    Attributes:
        documents_dir: Directory containing source documents.
        vector_store_dir: Production vector store (used only for reference).
        embedding_model: Embedding model name.
        embedding_device: Device for embeddings.
        chunk_sizes: Candidate chunk sizes to test.
        chunk_overlaps: Candidate chunk overlaps to test.
        top_ks: Candidate retrieval top-k values.
        sample_queries: Queries used to benchmark retrieval quality.
        results: List of per-configuration result records.
    """

    def __init__(
        self,
        documents_dir: str = "data/documents/",
        vector_store_dir: str = "data/vector_store/",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_device: Optional[str] = None,
        chunk_sizes: Optional[List[int]] = None,
        chunk_overlaps: Optional[List[int]] = None,
        top_ks: Optional[List[int]] = None,
        sample_queries: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the PerformanceEvaluator.

        Args:
            documents_dir: Directory with source documents to index.
            vector_store_dir: Production vector store directory (reference only).
            embedding_model: Sentence Transformer model name.
            embedding_device: Device for embeddings.
            chunk_sizes: Chunk sizes to sweep (default [300, 500, 800]).
            chunk_overlaps: Chunk overlaps to sweep (default [30, 50, 100]).
            top_ks: Top-k values to sweep (default [3, 5, 10]).
            sample_queries: Queries used for benchmarking.
        """
        self.documents_dir = documents_dir
        self.vector_store_dir = vector_store_dir
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device

        self.chunk_sizes = chunk_sizes or [300, 500, 800]
        self.chunk_overlaps = chunk_overlaps or [30, 50, 100]
        self.top_ks = top_ks or [3, 5, 10]

        self.sample_queries = sample_queries or [
            "What is CARIVIX AI?",
            "What machine learning algorithms are supported?",
            "How does the platform handle data preprocessing?",
            "What is economic analysis?",
            "Tell me about market prediction.",
        ]

        # Shared components (loaded once, reused across configs)
        self.loader = DocumentLoader(documents_dir=documents_dir)
        self.preprocessor = TextPreprocessor()
        self.embedding_generator = EmbeddingGenerator(
            model_name=embedding_model,
            device=embedding_device,
        )

        # Results
        self.results: List[Dict[str, Any]] = []
        self._best_config: Optional[Dict[str, Any]] = None

        logger.info(
            "PerformanceEvaluator initialized. Chunk sizes: %s, Overlaps: %s, "
            "Top-k: %s",
            self.chunk_sizes,
            self.chunk_overlaps,
            self.top_ks,
        )

    # ------------------------------------------------------------------
    # Document Preparation
    # ------------------------------------------------------------------

    def _load_documents(self) -> List[Document]:
        """Load and clean source documents once."""
        documents = self.loader.load_all()
        if not documents:
            raise RuntimeError(
                f"No documents found in '{self.documents_dir}'. "
                "Cannot run performance sweep."
            )
        return self.preprocessor.clean_documents(documents)

    # ------------------------------------------------------------------
    # Single Configuration Evaluation
    # ------------------------------------------------------------------

    def _evaluate_config(
        self,
        documents: List[Document],
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
    ) -> Dict[str, Any]:
        """
        Build an in-memory index for one configuration and benchmark it.

        Args:
            documents: Cleaned source documents.
            chunk_size: Chunk size for this configuration.
            chunk_overlap: Chunk overlap for this configuration.
            top_k: Retrieval top-k for this configuration.

        Returns:
            A record with timing and quality metrics.
        """
        record = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
            "chunk_count": 0,
            "embedding_time": 0.0,
            "indexing_time": 0.0,
            "retrieval_latency": 0.0,
            "total_query_time": 0.0,
            "avg_similarity": 0.0,
            "lexical_relevance": 0.0,
            "num_queries": len(self.sample_queries),
            "num_results": 0,
        }

        # ---- Split into chunks ----
        splitter = DocumentSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        record["chunk_count"] = len(chunks)

        if not chunks:
            logger.warning(
                "No chunks for config %d/%d. Skipping.", chunk_size, chunk_overlap
            )
            return record

        # ---- Generate embeddings ----
        with Timer("Embedding generation") as t:
            embeddings = self.embedding_generator.generate_from_documents(
                chunks, show_progress=False
            )
        record["embedding_time"] = round(t["elapsed"], 6)

        # ---- Build in-memory FAISS index (never persisted) ----
        store = VectorStore(index_directory=":memory:", index_type="L2")
        with Timer("FAISS indexing") as t:
            store.create_index(chunks, embeddings)
        record["indexing_time"] = round(t["elapsed"], 6)

        # ---- Retriever ----
        retriever = Retriever(
            embedding_generator=self.embedding_generator,
            vector_store=store,
        )

        # ---- Run queries ----
        retrieval_times = []
        query_times = []
        all_scores = []
        all_lexical = []
        total_results = 0

        for query in self.sample_queries:
            with Timer("Query") as tq:
                results = retriever.retrieve(query, k=top_k)
            query_times.append(tq["elapsed"])

            if results:
                retrieval_times.append(tq["elapsed"])
                total_results += len(results)
                all_scores.extend(r["score"] for r in results)
                all_lexical.extend(
                    lexical_overlap(query, r["document"].page_content)
                    for r in results
                )

        # ---- Aggregate metrics ----
        record["retrieval_latency"] = round(
            sum(retrieval_times) / len(retrieval_times), 6
        ) if retrieval_times else 0.0
        record["total_query_time"] = round(
            sum(query_times) / len(query_times), 6
        ) if query_times else 0.0
        record["avg_similarity"] = round(
            sum(all_scores) / len(all_scores), 6
        ) if all_scores else 0.0
        record["lexical_relevance"] = round(
            sum(all_lexical) / len(all_lexical), 6
        ) if all_lexical else 0.0
        record["num_results"] = total_results

        logger.info(
            "Config (%d/%d/k=%d) -> chunks=%d, embed=%.3fs, index=%.3fs, "
            "retrieval=%.4fs, avg_sim=%.4f, lexical=%.4f",
            chunk_size,
            chunk_overlap,
            top_k,
            record["chunk_count"],
            record["embedding_time"],
            record["indexing_time"],
            record["retrieval_latency"],
            record["avg_similarity"],
            record["lexical_relevance"],
        )
        return record

    # ------------------------------------------------------------------
    # Full Sweep
    # ------------------------------------------------------------------

    def run_sweep(self) -> List[Dict[str, Any]]:
        """
        Run the full configuration sweep across all chunk/overlap/top-k combos.

        Returns:
            List of per-configuration result records, sorted by a composite
            quality score (quality first, then speed).
        """
        logger.info("=" * 70)
        logger.info("  RETRIEVAL PERFORMANCE OPTIMIZATION SWEEP")
        logger.info("=" * 70)

        documents = self._load_documents()
        logger.info("Loaded %d cleaned documents.", len(documents))

        self.results = []

        for chunk_size in self.chunk_sizes:
            for chunk_overlap in self.chunk_overlaps:
                for top_k in self.top_ks:
                    record = self._evaluate_config(
                        documents,
                        chunk_size,
                        chunk_overlap,
                        top_k,
                    )
                    self.results.append(record)

        # Rank configurations.
        # Lower retrieval latency and higher relevance are better.
        # We use a composite score: quality (60%) + speed (40%).
        for record in self.results:
            if record["num_results"] == 0:
                record["composite_score"] = 0.0
                continue
            # Normalize speed: faster = higher score (relative to slowest).
            all_latencies = [r["retrieval_latency"] for r in self.results if r["retrieval_latency"] > 0]
            max_latency = max(all_latencies) if all_latencies else 1.0
            speed_score = 1.0 - (record["retrieval_latency"] / max_latency)
            quality_score = (record["avg_similarity"] + record["lexical_relevance"]) / 2.0
            record["composite_score"] = round(0.6 * quality_score + 0.4 * max(speed_score, 0.0), 6)

        # Sort by composite score (descending), then by retrieval speed.
        self.results.sort(
            key=lambda r: (
                r["composite_score"],
                -r["retrieval_latency"],
            ),
            reverse=True,
        )

        if self.results:
            self._best_config = self.results[0]

        logger.info(
            "Sweep complete. %d configurations evaluated. Best: %s",
            len(self.results),
            self._best_config,
        )
        return self.results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def best_config(self) -> Optional[Dict[str, Any]]:
        """Return the best configuration record."""
        return self._best_config

    def get_results_table(self) -> str:
        """
        Render the performance comparison table for the console.

        Returns:
            A formatted table of all configurations.
        """
        rows = []
        for r in self.results:
            rows.append([
                r["chunk_size"],
                r["chunk_overlap"],
                r["top_k"],
                r["chunk_count"],
                format_elapsed(r["embedding_time"]),
                format_elapsed(r["indexing_time"]),
                format_elapsed(r["retrieval_latency"]),
                format_elapsed(r["total_query_time"]),
                f"{r['avg_similarity']:.4f}",
                f"{r['lexical_relevance']:.4f}",
                f"{r['composite_score']:.4f}",
            ])
        return render_table(
            [
                "Chunk Size", "Overlap", "Top-k", "Chunks",
                "Embed Time", "Index Time", "Retrieval", "Total Query",
                "Avg Sim", "Rel.", "Score",
            ],
            rows,
            title="CONFIGURATION PERFORMANCE COMPARISON",
            max_width=160,
        )

    def get_best_config_table(self) -> str:
        """Render the best configuration as a table."""
        if not self._best_config:
            return "No best configuration available."
        b = self._best_config
        rows = [
            ["Chunk Size", b["chunk_size"]],
            ["Chunk Overlap", b["chunk_overlap"]],
            ["Top-k", b["top_k"]],
            ["Chunk Count", b["chunk_count"]],
            ["Embedding Time", format_elapsed(b["embedding_time"])],
            ["FAISS Indexing Time", format_elapsed(b["indexing_time"])],
            ["Retrieval Latency", format_elapsed(b["retrieval_latency"])],
            ["Total Query Time", format_elapsed(b["total_query_time"])],
            ["Average Similarity", f"{b['avg_similarity']:.4f}"],
            ["Lexical Relevance", f"{b['lexical_relevance']:.4f}"],
            ["Composite Score", f"{b['composite_score']:.4f}"],
        ]
        return render_table(["Parameter", "Value"], rows, title="BEST CONFIGURATION")

    def print_results(self) -> None:
        """Print all sweep results to the console."""
        print("\n" + self.get_results_table())
        print("\n" + self.get_best_config_table())

