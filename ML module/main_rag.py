#!/usr/bin/env python3
"""
CARIVIX AI - RAG Pipeline CLI Entry Point
===========================================

Command-line interface for the CARIVIX AI RAG (Retrieval-Augmented
Generation) pipeline.

Supports:
    - Indexing documents into a FAISS vector database
    - Querying the indexed documents with natural language questions
    - Running comprehensive tests
    - Displaying pipeline configuration and statistics

Usage:
    python main_rag.py --index                          # Index all documents
    python main_rag.py --query "What is CARIVIX AI?"    # Query indexed documents
    python main_rag.py --test                           # Run indexing test
    python main_rag.py --interactive                    # Interactive Q&A session
    python main_rag.py --info                           # Show pipeline info

Configuration:
    Edit the RAGPipeline initialization parameters below or pass
    CLI arguments to customize behavior.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logger, ensure_directory, get_timestamp
from rag.pipeline import RAGPipeline

# =============================================================================
# Constants
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DEFAULT_LOG_FILE = os.path.join(LOG_DIR, f"rag_{get_timestamp()}.log")

# Default paths
DEFAULT_DOCUMENTS_DIR = os.path.join(PROJECT_ROOT, "data", "documents")
DEFAULT_VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "data", "vector_store")

logger = logging.getLogger("CARIVIX_AI")

# =============================================================================
# Display Utilities
# =============================================================================

def print_header(title: str, char: str = "=", width: int = 70) -> None:
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")

def print_indexing_stats(stats: Dict[str, Any]) -> None:
    """Print indexing statistics in a readable format."""
    print_header("INDEXING STATISTICS")

    if not stats:
        print("  No indexing statistics available.")
        return

    print(f"  Documents Loaded:        {stats.get('documents_loaded', 0)}")
    print(f"  Chunks Created:          {stats.get('chunks_created', 0)}")
    print(f"  Embeddings Generated:    {stats.get('embeddings_generated', 0)}")
    print(f"  Embedding Dimension:     {stats.get('embedding_dimension', 0)}")
    print(f"  FAISS Vectors:           {stats.get('index_vector_count', 0)}")
    print(f"  Index Saved:             {stats.get('index_saved', False)}")
    print(f"  Index Path:              {stats.get('index_path', 'N/A')}")
    print(f"  Metadata Path:           {stats.get('metadata_path', 'N/A')}")
    print(f"  Elapsed Time:            {stats.get('elapsed_time_formatted', 'N/A')}")

    if "error" in stats:
        print(f"\n  ⚠ Error: {stats['error']}")

def print_query_result(result: Dict[str, Any]) -> None:
    """Print a formatted query result."""
    print_header("QUERY RESULT")

    print(f"\n  Question: {result['question']}")
    print(f"\n  Response: {result['response']}")

    print(f"\n  {'─' * 50}")
    print(f"  Retrieved Context:")

    for chunk in result.get("retrieved_chunks", []):
        score = chunk.get("score", 0)
        source = chunk.get("source", "unknown")
        text = chunk.get("text", chunk.get("document", ""))
        if hasattr(text, "page_content"):
            text = text.page_content
        text_preview = str(text)[:200]

        print(f"\n    [{score:.4f}] {source}")
        print(f"    {text_preview}...")

    print(f"\n  {'─' * 50}")
    print(f"  Retrieval Time:  {result.get('retrieval_time', 0):.4f}s")
    print(f"  Generation Time: {result.get('generation_time', 0):.4f}s")
    print(f"  Total Time:      {result.get('total_time', 0):.4f}s")
    print(f"{'=' * 70}")

def print_pipeline_info(pipeline: RAGPipeline) -> None:
    """Print comprehensive pipeline configuration information."""
    info = pipeline.get_pipeline_info()

    print_header("RAG PIPELINE CONFIGURATION")

    print(f"\n  Pipeline:")
    print(f"    Documents Directory:  {info['pipeline']['documents_dir']}")
    print(f"    Vector Store:         {info['pipeline']['vector_store_dir']}")
    print(f"    Indexed:              {'✓ Yes' if info['pipeline']['is_indexed'] else '✗ No'}")

    print(f"\n  Embedding:")
    print(f"    Model:                {info['embedding']['model_name']}")
    print(f"    Dimension:            {info['embedding']['dimension']}")
    print(f"    Device:               {info['embedding']['device']}")

    print(f"\n  Chunking:")
    print(f"    Chunk Size:           {info['chunking']['chunk_size']}")
    print(f"    Chunk Overlap:        {info['chunking']['chunk_overlap']}")

    print(f"\n  Retrieval:")
    print(f"    Top-K:                {info['retrieval']['k']}")

    print(f"\n  LLM:")
    print(f"    Backend:              {info['llm']['backend']}")
    print(f"    Model:                {info['llm']['model_name']}")
    print(f"    Available:            {'✓ Yes' if info['llm']['available'] else '✗ No'}")

    # Print indexing stats if available
    if info['pipeline']['is_indexed']:
        print(f"\n  {'─' * 50}")
        print(f"  Indexing Details:")
        stats = info['pipeline']['indexing_stats']
        for key, value in stats.items():
            if not key.startswith("_"):
                print(f"    {key.replace('_', ' ').title():<25} {value}")

# =============================================================================
# CLI Actions
# =============================================================================

def action_index(
    pipeline: RAGPipeline,
    force: bool = False,
) -> None:
    """
    Index all documents in the configured directory.

    Args:
        pipeline: Configured RAGPipeline instance.
        force: If True, re-index even if already indexed.
    """
    print_header("INDEXING DOCUMENTS")
    print(f"  Documents directory: {pipeline.documents_dir}")
    print(f"  Vector store:        {pipeline.vector_store_dir}")

    stats = pipeline.index_documents(force_reindex=force)
    print_indexing_stats(stats)

def action_query(
    pipeline: RAGPipeline,
    question: str,
    k: Optional[int] = None,
    verbose: bool = True,
) -> None:
    """
    Query the indexed documents with a natural language question.

    Args:
        pipeline: Configured RAGPipeline instance.
        question: User's question.
        k: Number of chunks to retrieve.
        verbose: Whether to show detailed output.
    """
    # Ensure index exists
    if not pipeline._is_indexed:
        if not pipeline.load_index():
            print("\n  ⚠ No index found. Indexing documents first...")
            pipeline.index_documents()

    print_header("PROCESSING QUERY")
    print(f"  Question: {question}")

    try:
        result = pipeline.query(
            question=question,
            k=k,
            verbose=verbose,
        )
        print_query_result(result)
        return result
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        print(f"\n  ❌ Query failed: {exc}")
        return None

def action_test(pipeline: RAGPipeline) -> None:
    """
    Run the indexing test: load, chunk, embed, index, and display statistics.

    Args:
        pipeline: Configured RAGPipeline instance.
    """
    print_header("RAG PIPELINE INDEXING TEST")

    # Index documents
    print("\n  Step 1: Indexing documents...")
    stats = pipeline.index_documents(force_reindex=True)

    if stats.get("error"):
        print(f"\n  ❌ Indexing failed: {stats['error']}")
        return

    print_indexing_stats(stats)

    # Test search with sample queries
    print("\n" + "=" * 70)
    print("  Step 2: Testing similarity search...")
    print("=" * 70)

    test_queries = [
        "What is CARIVIX AI?",
        "What machine learning algorithms are supported?",
        "How does the platform handle data?",
        "What is economic analysis?",
        "Tell me about market prediction",
    ]

    for query in test_queries:
        result = pipeline.query(query, verbose=False)
        chunks = result.get("retrieved_chunks", [])
        if chunks:
            top_score = chunks[0].get("score", 0)
            print(f"  ✓ '{query[:50]}...' → Top score: {top_score:.4f}")
        else:
            print(f"  ⚠ '{query[:50]}...' → No results")

    # Summary
    print("\n" + "=" * 70)
    print("  TEST COMPLETE")
    print("=" * 70)
    print(f"  Documents indexed: {stats.get('documents_loaded', 0)}")
    print(f"  Total chunks:      {stats.get('chunks_created', 0)}")
    print(f"  Test queries run:  {len(test_queries)}")

def action_interactive(pipeline: RAGPipeline) -> None:
    """
    Start an interactive Q&A session.

    Args:
        pipeline: Configured RAGPipeline instance.
    """
    # Ensure index exists
    if not pipeline._is_indexed:
        if not pipeline.load_index():
            print("\n  No index found. Indexing documents first...")
            pipeline.index_documents()

    print_header("INTERACTIVE Q&A SESSION")
    print("  Type 'exit', 'quit', or 'q' to end the session.")
    print("  Type 'info' to see pipeline configuration.")
    print("  Type 'reindex' to force re-index documents.")
    print("=" * 70)

    while True:
        try:
            question = input("\n  Question: ").strip()

            if question.lower() in ("exit", "quit", "q"):
                print("\n  Goodbye!")
                break

            if question.lower() == "info":
                print_pipeline_info(pipeline)
                continue

            if question.lower() == "reindex":
                pipeline.index_documents(force_reindex=True)
                print("  ✓ Re-indexing complete.")
                continue

            if not question:
                continue

            result = pipeline.query(question, verbose=False)
            print(f"\n  Answer: {result['response']}")
            print(f"\n  (Retrieved {len(result['retrieved_chunks'])} chunks "
                  f"in {result['retrieval_time']:.4f}s, "
                  f"generated in {result['generation_time']:.4f}s)")

        except KeyboardInterrupt:
            print("\n\n  Goodbye!")
            break
        except Exception as exc:
            logger.error("Interactive session error: %s", exc)
            print(f"\n  ⚠ Error: {exc}")

def action_info(pipeline: RAGPipeline, show_full: bool = False) -> None:
    """
    Display pipeline configuration and statistics.

    Args:
        pipeline: Configured RAGPipeline instance.
        show_full: If True, show full detailed output.
    """
    print_pipeline_info(pipeline)

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main CLI entry point for the RAG pipeline."""
    parser = argparse.ArgumentParser(
        description="CARIVIX AI - Retrieval-Augmented Generation (RAG) Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_rag.py --index                         # Index all documents
  python main_rag.py --query "What is CARIVIX AI?"   # Query documents
  python main_rag.py --test                           # Run indexing test
  python main_rag.py --interactive                    # Interactive Q&A
  python main_rag.py --info                           # Show config

  # With custom paths:
  python main_rag.py --index \\
      --documents data/my_docs/ \\
      --vector-store data/my_vector_store/

  # With custom LLM:
  python main_rag.py --query "Question" \\
      --llm-backend ollama --llm-model llama3.1
        """,
    )

    # Actions
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index all documents into the vector database",
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        metavar="QUESTION",
        help="Query the indexed documents with a question",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run comprehensive indexing and retrieval test",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive Q&A session",
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Show pipeline configuration and statistics",
    )

    # Pipeline configuration
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
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in characters (default: 500)",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in characters (default: 50)",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2)",
    )

    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )

    # LLM Configuration
    parser.add_argument(
        "--llm-backend",
        type=str,
        default="ollama",
        choices=["ollama", "huggingface"],
        help="LLM backend (default: ollama)",
    )

    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model name (default depends on backend)",
    )

    parser.add_argument(
        "--llm-url",
        type=str,
        default=None,
        help="Ollama server URL (default: http://localhost:11434)",
    )

    # Other options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if already indexed",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save query result to JSON file",
    )

    args = parser.parse_args()

    # --- Setup Logging ---
    ensure_directory(LOG_DIR)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(
        name="CARIVIX_AI",
        log_file=DEFAULT_LOG_FILE,
        level=log_level,
    )

    logger.info("=" * 70)
    logger.info("  CARIVIX AI - RAG PIPELINE")
    logger.info("  Started at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 70)

    # --- Build LLM kwargs ---
    llm_kwargs = {}
    if args.llm_url:
        llm_kwargs["base_url"] = args.llm_url

    # --- Initialize Pipeline ---
    logger.info("Initializing RAG pipeline...")
    pipeline = RAGPipeline(
        documents_dir=args.documents,
        vector_store_dir=args.vector_store,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        llm_kwargs=llm_kwargs if llm_kwargs else None,
        retrieval_k=args.retrieval_k,
    )

    # --- Determine Action ---
    actions_taken = 0

    if args.index:
        actions_taken += 1
        action_index(pipeline, force=args.force)

    if args.query:
        actions_taken += 1
        result = action_query(pipeline, args.query, k=args.retrieval_k)

        # Save to JSON if requested
        if args.output and result:
            output_path = os.path.join(PROJECT_ROOT, args.output)
            ensure_directory(os.path.dirname(output_path))
            with open(output_path, "w", encoding="utf-8") as f:
                # Convert non-serializable objects
                serializable = {
                    "question": result["question"],
                    "response": result["response"],
                    "retrieval_time": result.get("retrieval_time"),
                    "generation_time": result.get("generation_time"),
                    "total_time": result.get("total_time"),
                    "retrieved_chunks": [
                        {
                            "rank": c.get("rank"),
                            "score": c.get("score"),
                            "source": c.get("source"),
                            "page": c.get("page"),
                            "chunk_id": c.get("chunk_id"),
                            "text_preview": str(c.get("text", ""))[:300],
                        }
                        for c in result.get("retrieved_chunks", [])
                    ],
                }
                json.dump(serializable, f, indent=2)
            print(f"\n  Results saved to: {output_path}")

    if args.test:
        actions_taken += 1
        action_test(pipeline)

    if args.interactive:
        actions_taken += 1
        action_interactive(pipeline)

    if args.info:
        actions_taken += 1
        action_info(pipeline)

    if actions_taken == 0:
        # Default: show info and suggest actions
        print_header("CARIVIX AI - RAG PIPELINE")
        print("\n  No action specified. Use one of the following flags:")
        print("\n    python main_rag.py --index           # Index documents")
        print("    python main_rag.py --query \"...\"    # Query documents")
        print("    python main_rag.py --test            # Run tests")
        print("    python main_rag.py --interactive     # Interactive session")
        print("    python main_rag.py --info            # Show configuration")
        print("\n  Or use --help for full documentation.\n")

        # Show basic info
        action_info(pipeline)

    logger.info("=" * 70)
    logger.info("  RAG PIPELINE COMPLETED")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()


