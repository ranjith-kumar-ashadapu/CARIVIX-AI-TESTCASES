#!/usr/bin/env python3
"""
CARIVIX AI - RAG Pipeline Evaluation & Optimization CLI
=========================================================

Command-line interface for evaluating the existing CARIVIX AI RAG pipeline
and optimizing its retrieval configuration. This module does NOT modify the
existing pipeline implementation.

Modes:
    all          Evaluate vector DB integration AND run the optimization sweep
    vector-db    Evaluate the FAISS vector database integration only
    optimize     Run the retrieval performance configuration sweep only

Usage:
    python evaluate_rag.py --mode all
    python evaluate_rag.py --mode vector-db
    python evaluate_rag.py --mode optimize
    python evaluate_rag.py --mode all --output reports/evaluation_report.md
    python evaluate_rag.py --mode all --queries "What is CARIVIX AI?" "Tell me about economics"
"""

import os
import sys
import logging
import argparse
from typing import Any, Dict, List, Optional

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logger, ensure_directory, get_timestamp
from rag.evaluation import (
    VectorDBEvaluator,
    PerformanceEvaluator,
    EvaluationReport,
)

# =============================================================================
# Constants
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DEFAULT_REPORT_PATH = os.path.join(PROJECT_ROOT, "evaluation_report.md")

DEFAULT_DOCUMENTS_DIR = os.path.join(PROJECT_ROOT, "data", "documents")
DEFAULT_VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "data", "vector_store")

logger = logging.getLogger("CARIVIX_AI")

# =============================================================================
# Evaluation Orchestration
# =============================================================================

def collect_system_config() -> Dict[str, Any]:
    """
    Return the current system configuration summary.

    Mirrors the user's stated configuration for the existing pipeline.
    """
    return {
        "Document Loader": "Implemented (PDF/DOCX/TXT/CSV)",
        "Text Preprocessing": "Enabled",
        "Chunk Size": 500,
        "Chunk Overlap": 50,
        "Embedding Model": "sentence-transformers/all-MiniLM-L6-v2",
        "Embedding Dimension": 384,
        "Vector Database": "FAISS",
        "Retrieval Top-k": 5,
        "LLM Runtime": "Ollama",
        "Processing Device": "CPU",
    }

def run_vector_db_evaluation(args) -> Dict[str, Any]:
    """
    Run the FAISS vector database integration evaluation.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The evaluation summary dictionary.
    """
    print("\n" + "=" * 70)
    print("  STEP 1: VECTOR DATABASE INTEGRATION EVALUATION")
    print("=" * 70)

    queries = args.queries or None
    evaluator = VectorDBEvaluator(
        vector_store_dir=args.vector_store,
        embedding_model=args.embedding_model,
        embedding_device=args.device,
        sample_queries=queries,
        top_k=args.top_k,
    )

    summary = evaluator.run_evaluation()
    evaluator.print_summary()
    return summary

def run_optimization_sweep(args) -> List[Dict[str, Any]]:
    """
    Run the retrieval performance configuration sweep.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The list of per-configuration result records.
    """
    print("\n" + "=" * 70)
    print("  STEP 2: RETRIEVAL PERFORMANCE OPTIMIZATION SWEEP")
    print("=" * 70)

    queries = args.queries or None
    evaluator = PerformanceEvaluator(
        documents_dir=args.documents,
        vector_store_dir=args.vector_store,
        embedding_model=args.embedding_model,
        embedding_device=args.device,
        chunk_sizes=args.chunk_sizes,
        chunk_overlaps=args.chunk_overlaps,
        top_ks=args.top_ks_list,
        sample_queries=queries,
    )

    results = evaluator.run_sweep()
    evaluator.print_results()

    best = evaluator.best_config
    if best:
        print("\n" + "=" * 70)
        print("  RECOMMENDED BEST CONFIGURATION")
        print("=" * 70)
        print(
            f"  Chunk Size:       {best['chunk_size']}\n"
            f"  Chunk Overlap:    {best['chunk_overlap']}\n"
            f"  Retrieval Top-k:  {best['top_k']}\n"
            f"  Retrieval Latency: {best['retrieval_latency']:.4f}s\n"
            f"  Avg Similarity:   {best['avg_similarity']:.4f}\n"
            f"  Composite Score:  {best['composite_score']:.4f}"
        )
        print("=" * 70)

    return results

# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main CLI entry point for the evaluation module."""
    parser = argparse.ArgumentParser(
        description="CARIVIX AI - RAG Pipeline Evaluation & Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_rag.py --mode all
  python evaluate_rag.py --mode vector-db
  python evaluate_rag.py --mode optimize
  python evaluate_rag.py --mode all --output reports/evaluation_report.md
  python evaluate_rag.py --mode all --chunk-sizes 300 500 800
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "vector-db", "optimize"],
        help="Evaluation mode (default: all)",
    )

    parser.add_argument(
        "--documents",
        type=str,
        default=DEFAULT_DOCUMENTS_DIR,
        help=f"Documents directory (default: {DEFAULT_DOCUMENTS_DIR})",
    )

    parser.add_argument(
        "--vector-store",
        type=str,
        default=DEFAULT_VECTOR_STORE_DIR,
        help=f"Vector store directory (default: {DEFAULT_VECTOR_STORE_DIR})",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Embedding device (default: auto-detect)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k for vector DB evaluation (default: 5)",
    )

    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[300, 500, 800],
        help="Chunk sizes to sweep (default: 300 500 800)",
    )

    parser.add_argument(
        "--chunk-overlaps",
        type=int,
        nargs="+",
        default=[30, 50, 100],
        help="Chunk overlaps to sweep (default: 30 50 100)",
    )

    parser.add_argument(
        "--top-ks-list",
        type=int,
        nargs="+",
        default=[3, 5, 10],
        help="Top-k values to sweep (default: 3 5 10)",
    )

    parser.add_argument(
        "--queries",
        type=str,
        nargs="+",
        default=None,
        help="Custom sample queries for evaluation",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_REPORT_PATH,
        help="Report output path (default: evaluation_report.md)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    # --- Setup Logging ---
    ensure_directory(LOG_DIR)
    log_file = os.path.join(LOG_DIR, f"evaluate_rag_{get_timestamp()}.log")
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(name="CARIVIX_AI", log_file=log_file, level=log_level)

    logger.info("=" * 70)
    logger.info("  CARIVIX AI - RAG EVALUATION & OPTIMIZATION")
    logger.info("  Mode: %s", args.mode)
    logger.info("=" * 70)

    # --- Run evaluations ---
    vector_db_summary: Dict[str, Any] = {}
    performance_results: List[Dict[str, Any]] = []
    best_config: Optional[Dict[str, Any]] = None

    try:
        if args.mode in ("all", "vector-db"):
            vector_db_summary = run_vector_db_evaluation(args)

        if args.mode in ("all", "optimize"):
            performance_results = run_optimization_sweep(args)
            if performance_results:
                best_config = performance_results[0]

        # --- Generate report ---
        system_config = collect_system_config()
        report = EvaluationReport(
            vector_db_summary=vector_db_summary,
            performance_results=performance_results,
            best_config=best_config,
            system_config=system_config,
        )

        report.print_console_report()
        report_path = report.save_markdown_report(args.output)

        print("\n" + "=" * 70)
        print(f"  EVALUATION COMPLETE")
        print(f"  Report saved to: {report_path}")
        print(f"  Log file: {log_file}")
        print("=" * 70 + "\n")

    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        print(f"\n  ❌ Evaluation failed: {exc}")
        raise

if __name__ == "__main__":
    main()

