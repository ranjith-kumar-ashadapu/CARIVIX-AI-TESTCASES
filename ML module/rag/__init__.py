"""
CARIVIX AI - Retrieval-Augmented Generation (RAG) Pipeline
============================================================

A modular, production-style RAG pipeline for the CARIVIX AI project.

Pipeline Flow:
    Load Documents
        ↓
    Preprocess & Split into Chunks
        ↓
    Generate Embeddings (all-MiniLM-L6-v2)
        ↓
    Create FAISS Vector Index
        ↓
    Save Index Locally
        ↓
    User Query → Retrieve Top-5 Chunks → Build Prompt → LLM → Response

Modules:
    loader          - Document loading (PDF, DOCX, TXT, CSV)
    splitter        - Text preprocessing and semantic chunking
    embeddings      - Embedding generation with Sentence Transformers
    vector_store    - FAISS vector database operations
    retriever       - Similarity search and retrieval
    prompt_builder  - Context-aware prompt construction
    generator       - LLM response generation (Ollama / HuggingFace)
    pipeline        - End-to-end RAG pipeline orchestrator
"""

__version__ = "1.0.0"
__author__ = "CARIVIX AI"

try:
    from rag.config import DEFAULT_RAG_CONFIG, RAGConfig, get_rag_config
except ImportError:
    pass

try:
    from rag.context_builder import ContextBuilder
except ImportError:
    pass

try:
    from rag.loader import DocumentLoader
except ImportError:
    pass

try:
    from rag.splitter import TextPreprocessor, DocumentSplitter
except ImportError:
    pass

try:
    from rag.embeddings import EmbeddingGenerator
except ImportError:
    pass

try:
    from rag.vector_store import VectorStore
except ImportError:
    pass

try:
    from rag.retriever import Retriever
except ImportError:
    pass

try:
    from rag.prompt_builder import PromptBuilder
except ImportError:
    pass

try:
    from rag.generator import ResponseGenerator
except ImportError:
    pass

try:
    from rag.pipeline import RAGPipeline
except ImportError:
    pass

try:
    from rag.evaluation.relevance_evaluator import RelevanceEvaluator
    from rag.evaluation.factuality_evaluator import FactualityEvaluator
    from rag.evaluation.test_set_generator import TestSetGenerator, EvaluationCase
except ImportError:
    pass

__all__ = [
    "RAGConfig",
    "DEFAULT_RAG_CONFIG",
    "get_rag_config",
    "ContextBuilder",
    "DocumentLoader",
    "TextPreprocessor",
    "DocumentSplitter",
    "EmbeddingGenerator",
    "VectorStore",
    "Retriever",
    "PromptBuilder",
    "ResponseGenerator",
    "RAGPipeline",
    "RelevanceEvaluator",
    "FactualityEvaluator",
    "TestSetGenerator",
    "EvaluationCase",
]

