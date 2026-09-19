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

from rag.config import DEFAULT_RAG_CONFIG, RAGConfig, get_rag_config
from rag.context_builder import ContextBuilder
from rag.loader import DocumentLoader
from rag.splitter import TextPreprocessor, DocumentSplitter
from rag.embeddings import EmbeddingGenerator
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.prompt_builder import PromptBuilder
from rag.generator import ResponseGenerator
from rag.pipeline import RAGPipeline
from rag.evaluation.relevance_evaluator import RelevanceEvaluator
from rag.evaluation.factuality_evaluator import FactualityEvaluator
from rag.evaluation.test_set_generator import TestSetGenerator, EvaluationCase

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

