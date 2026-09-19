"""
CARIVIX AI - RAG Pipeline Evaluation & Optimization Module
============================================================

Production-grade evaluation and optimization toolkit for the existing
CARIVIX AI RAG pipeline. This module does NOT modify the core pipeline;
it only reuses the public components (EmbeddingGenerator, VectorStore,
Retriever, RAGPipeline, etc.) to:

    1. Evaluate the FAISS vector database integration.
    2. Benchmark retrieval performance across multiple configurations
       (chunk size, chunk overlap, top-k).
    3. Generate a comprehensive Markdown report (evaluation_report.md).

Modules:
    utils                  - Shared helpers (timing, tables, scoring)
    vector_db_evaluator    - FAISS index validation & similarity search audit
    performance_evaluator  - Configuration sweep / latency benchmarking
    report                 - Markdown & console report generation

Usage:
    python evaluate_rag.py --mode all
    python evaluate_rag.py --mode vector-db
    python evaluate_rag.py --mode optimize
"""

from rag.evaluation.utils import Timer, format_elapsed, render_table
from rag.evaluation.vector_db_evaluator import VectorDBEvaluator
from rag.evaluation.performance_evaluator import PerformanceEvaluator
from rag.evaluation.report import EvaluationReport
from rag.evaluation.test_set_generator import TestSetGenerator, EvaluationCase
from rag.evaluation.relevance_evaluator import RelevanceEvaluator
from rag.evaluation.factuality_evaluator import FactualityEvaluator

__version__ = "1.0.0"
__author__ = "CARIVIX AI"

__all__ = [
    "Timer",
    "format_elapsed",
    "render_table",
    "VectorDBEvaluator",
    "PerformanceEvaluator",
    "EvaluationReport",
    "TestSetGenerator",
    "EvaluationCase",
    "RelevanceEvaluator",
    "FactualityEvaluator",
]

