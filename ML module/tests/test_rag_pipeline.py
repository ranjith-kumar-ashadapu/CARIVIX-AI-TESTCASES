"""
RAG Pipeline Integration Tests for CARIVIX AI
===============================================

Tests for the complete RAG pipeline:
    - Document loading (all formats)
    - Text preprocessing and chunking
    - Embedding generation
    - FAISS index creation, saving, and loading
    - Similarity search and retrieval
    - Prompt building
    - End-to-end indexing pipeline

Usage:
    python -m pytest tests/test_rag_pipeline.py -v
    python -m pytest tests/test_rag_pipeline.py::test_document_loading -v
"""

import os
import sys
import tempfile
import logging
from typing import List

import numpy as np
import pytest

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.loader import DocumentLoader
from rag.splitter import TextPreprocessor, DocumentSplitter
from rag.embeddings import EmbeddingGenerator
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.prompt_builder import PromptBuilder
from rag.pipeline import RAGPipeline

# Disable logging during tests
logging.disable(logging.CRITICAL)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_text() -> str:
    """Provide a sample text for testing."""
    return (
        "CARIVIX AI is a comprehensive artificial intelligence platform "
        "designed for economic analysis. It integrates machine learning "
        "models and data processing pipelines. The platform supports "
        "multiple ML algorithms including Random Forest and XGBoost. "
        "Experiment tracking is done with MLflow."
    )

@pytest.fixture
def sample_documents_dir(tmpdir) -> str:
    """Create a temporary directory with sample documents."""
    docs_dir = os.path.join(tmpdir, "documents")
    os.makedirs(docs_dir, exist_ok=True)

    # Create a TXT file
    with open(os.path.join(docs_dir, "test.txt"), "w", encoding="utf-8") as f:
        f.write(
            "CARIVIX AI is a platform for economic analysis. "
            "It uses machine learning for market prediction. "
            "The platform handles data preprocessing and model training."
        )

    # Create a CSV file
    import pandas as pd
    df = pd.DataFrame({
        "feature": ["accuracy", "precision", "recall"],
        "value": [0.95, 0.93, 0.91],
        "description": ["Overall accuracy score", "Positive predictive value", "Sensitivity measure"],
    })
    df.to_csv(os.path.join(docs_dir, "metrics.csv"), index=False)

    return docs_dir

# =============================================================================
# Test 1: Document Loading
# =============================================================================

class TestDocumentLoader:
    """Test suite for DocumentLoader."""

    def test_load_txt(self, sample_documents_dir):
        """Test loading a TXT file."""
        loader = DocumentLoader(documents_dir=sample_documents_dir)
        documents = loader.load_all()
        assert len(documents) >= 1
        txt_docs = [d for d in documents if d.metadata.get("file_type") == "txt"]
        assert len(txt_docs) >= 1
        assert "CARIVIX AI" in txt_docs[0].page_content

    def test_load_csv(self, sample_documents_dir):
        """Test loading a CSV file."""
        loader = DocumentLoader(documents_dir=sample_documents_dir)
        documents = loader.load_all()
        csv_docs = [d for d in documents if d.metadata.get("file_type") == "csv"]
        assert len(csv_docs) >= 1
        assert "feature" in csv_docs[0].page_content or "accuracy" in csv_docs[0].page_content

    def test_load_nonexistent_directory(self):
        """Test loading from a non-existent directory."""
        loader = DocumentLoader(documents_dir="/nonexistent/path/")
        documents = loader.load_all()
        assert len(documents) == 0

    def test_count_files(self, sample_documents_dir):
        """Test file counting."""
        loader = DocumentLoader(documents_dir=sample_documents_dir)
        count = loader.count_files()
        assert count >= 2

    def test_supported_extensions(self):
        """Test supported extensions list."""
        loader = DocumentLoader()
        extensions = loader.get_supported_extensions()
        assert ".txt" in extensions
        assert ".csv" in extensions
        assert ".pdf" in extensions
        assert ".docx" in extensions

# =============================================================================
# Test 2: Text Preprocessing
# =============================================================================

class TestTextPreprocessor:
    """Test suite for TextPreprocessor."""

    def test_clean_text_removes_extra_spaces(self):
        """Test removal of extra whitespace."""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("  Hello   World  ")
        assert result == "Hello World"

    def test_clean_text_normalizes_newlines(self):
        """Test newline normalization."""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("Line1\r\nLine2\rLine3")
        assert "Line1\nLine2\nLine3" == result

    def test_clean_text_collapses_blank_lines(self):
        """Test collapsing multiple blank lines."""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("Para1\n\n\n\nPara2")
        assert "Para1\n\nPara2" == result

    def test_clean_text_removes_control_chars(self):
        """Test removal of control characters."""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("Hello\x00World\x01Test")
        assert "HelloWorldTest" == result

    def test_clean_documents(self, sample_text):
        """Test cleaning a list of documents."""
        from langchain_core.documents import Document
        preprocessor = TextPreprocessor()
        docs = [Document(page_content=sample_text)]
        cleaned = preprocessor.clean_documents(docs)
        assert len(cleaned) == 1
        assert "CARIVIX AI" in cleaned[0].page_content

    def test_word_count(self):
        """Test word counting."""
        count = TextPreprocessor.word_count("Hello world test")
        assert count == 3

    def test_empty_text(self):
        """Test cleaning empty text."""
        preprocessor = TextPreprocessor()
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text(None) == ""

# =============================================================================
# Test 3: Document Splitting
# =============================================================================

class TestDocumentSplitter:
    """Test suite for DocumentSplitter."""

    def test_split_documents_creates_chunks(self, sample_text):
        """Test splitting creates smaller chunks."""
        from langchain_core.documents import Document
        splitter = DocumentSplitter(chunk_size=100, chunk_overlap=10)
        docs = [Document(page_content=sample_text)]
        chunks = splitter.split_documents(docs)
        assert len(chunks) > 1

    def test_split_documents_metadata(self, sample_text):
        """Test chunk metadata."""
        from langchain_core.documents import Document
        splitter = DocumentSplitter(chunk_size=100, chunk_overlap=10)
        docs = [Document(page_content=sample_text, metadata={"source": "test.txt"})]
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert "chunk_total" in chunk.metadata
            assert chunk.metadata["source"] == "test.txt"

    def test_split_empty_list(self):
        """Test splitting empty document list."""
        splitter = DocumentSplitter()
        chunks = splitter.split_documents([])
        assert len(chunks) == 0

    def test_split_text(self, sample_text):
        """Test splitting raw text."""
        splitter = DocumentSplitter(chunk_size=200, chunk_overlap=20)
        chunks = splitter.split_text(sample_text, source="test.txt")
        assert len(chunks) >= 1

    def test_chunk_config(self):
        """Test chunk configuration."""
        splitter = DocumentSplitter(chunk_size=500, chunk_overlap=50)
        config = splitter.config
        assert config["chunk_size"] == 500
        assert config["chunk_overlap"] == 50

# =============================================================================
# Test 4: Embedding Generation
# =============================================================================

class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator."""

    @pytest.fixture
    def generator(self):
        """Provide an embedding generator instance."""
        return EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

    def test_generate_returns_numpy_array(self, generator):
        """Test embedding output type."""
        embeddings = generator.generate(["Hello world"])
        assert isinstance(embeddings, np.ndarray)

    def test_generate_dimension(self, generator):
        """Test embedding dimension."""
        embeddings = generator.generate(["Hello world"])
        assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension

    def test_generate_multiple_texts(self, generator):
        """Test batch embedding."""
        texts = ["Hello world", "CARIVIX AI", "Test sentence"]
        embeddings = generator.generate(texts)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 384

    def test_generate_single(self, generator):
        """Test single text embedding."""
        vector = generator.generate_single("Test query")
        assert vector.shape == (384,)

    def test_generate_empty_list_raises(self, generator):
        """Test empty list raises error."""
        with pytest.raises(ValueError):
            generator.generate([])

    def test_generate_empty_string_raises(self, generator):
        """Test empty string raises error."""
        with pytest.raises(ValueError):
            generator.generate_single("")

    def test_get_dimension(self, generator):
        """Test dimension retrieval."""
        dim = generator.get_dimension()
        assert dim == 384

    def test_get_model_info(self, generator):
        """Test model info."""
        info = generator.get_model_info()
        assert info["model_name"] == "all-MiniLM-L6-v2"
        assert info["dimension"] == 384

# =============================================================================
# Test 5: Vector Store (FAISS)
# =============================================================================

class TestVectorStore:
    """Test suite for VectorStore."""

    @pytest.fixture
    def chunks_and_embeddings(self):
        """Provide sample chunks and embeddings."""
        from langchain_core.documents import Document
        chunks = [
            Document(page_content="CARIVIX AI platform overview.", metadata={"source": "test.txt", "chunk_id": 1}),
            Document(page_content="Machine learning models for prediction.", metadata={"source": "test.txt", "chunk_id": 2}),
            Document(page_content="Data preprocessing and feature engineering.", metadata={"source": "test.txt", "chunk_id": 3}),
        ]
        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        embeddings = generator.generate_from_documents(chunks)
        return chunks, embeddings

    def test_create_index(self, chunks_and_embeddings):
        """Test index creation."""
        store = VectorStore(index_type="L2")
        chunks, embeddings = chunks_and_embeddings
        store.create_index(chunks, embeddings)
        assert store.is_index_loaded
        assert store.vector_count == len(chunks)

    def test_save_and_load_index(self, chunks_and_embeddings, tmpdir):
        """Test saving and loading index."""
        store = VectorStore(index_type="L2")
        chunks, embeddings = chunks_and_embeddings
        store.create_index(chunks, embeddings)

        save_dir = os.path.join(tmpdir, "vector_store")
        store.save_index(directory=save_dir)

        # Verify files exist
        assert os.path.exists(os.path.join(save_dir, "index.faiss"))
        assert os.path.exists(os.path.join(save_dir, "index.pkl"))

        # Load into a new store
        store2 = VectorStore(index_directory=save_dir)
        store2.load_index()
        assert store2.vector_count == len(chunks)

    def test_similarity_search(self, chunks_and_embeddings):
        """Test similarity search."""
        store = VectorStore(index_type="L2")
        chunks, embeddings = chunks_and_embeddings
        store.create_index(chunks, embeddings)

        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        query_vector = generator.generate_single("What is CARIVIX AI?")

        results = store.similarity_search(query_vector, k=2)
        assert len(results) == 2
        assert results[0]["score"] > 0

    def test_create_index_empty_raises(self):
        """Test creating index with empty data raises error."""
        store = VectorStore()
        with pytest.raises(ValueError):
            store.create_index([], np.array([]))

    def test_vector_count(self, chunks_and_embeddings):
        """Test vector count property."""
        store = VectorStore()
        assert store.vector_count == 0  # No index yet

        chunks, embeddings = chunks_and_embeddings
        store.create_index(chunks, embeddings)
        assert store.vector_count == len(chunks)

# =============================================================================
# Test 6: Retriever
# =============================================================================

class TestRetriever:
    """Test suite for Retriever."""

    @pytest.fixture
    def retriever(self):
        """Provide a configured retriever."""
        from langchain_core.documents import Document

        # Create test data
        chunks = [
            Document(page_content="CARIVIX AI is an AI platform for economic analysis.", metadata={"source": "test.txt", "chunk_id": 1}),
            Document(page_content="It uses Random Forest and XGBoost for predictions.", metadata={"source": "test.txt", "chunk_id": 2}),
            Document(page_content="MLflow is used for experiment tracking.", metadata={"source": "test.txt", "chunk_id": 3}),
        ]

        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        embeddings = generator.generate_from_documents(chunks)

        store = VectorStore()
        store.create_index(chunks, embeddings)

        return Retriever(embedding_generator=generator, vector_store=store)

    def test_retrieve_returns_results(self, retriever):
        """Test retrieval returns results."""
        results = retriever.retrieve("What is CARIVIX AI?", k=2)
        assert len(results) >= 1
        assert "score" in results[0]
        assert "document" in results[0]
        assert "source" in results[0]

    def test_retrieve_top_k(self, retriever):
        """Test top-k retrieval."""
        results = retriever.retrieve("machine learning", k=3)
        assert len(results) <= 3

    def test_retrieve_empty_query(self, retriever):
        """Test empty query returns empty."""
        results = retriever.retrieve("")
        assert len(results) == 0

    def test_format_results(self, retriever):
        """Test result formatting."""
        results = retriever.retrieve("CARIVIX", k=2)
        formatted = retriever.format_results(results)
        assert "CARIVIX" in formatted or "Score" in formatted

# =============================================================================
# Test 7: Prompt Builder
# =============================================================================

class TestPromptBuilder:
    """Test suite for PromptBuilder."""

    def test_build_prompt(self):
        """Test basic prompt building."""
        builder = PromptBuilder()
        chunks = [
            {"text": "CARIVIX AI is an AI platform.", "source": "test.txt", "score": 0.95, "chunk_id": 1},
        ]
        prompt = builder.build_prompt("What is CARIVIX?", chunks)
        assert "CARIVIX AI" in prompt
        assert "What is CARIVIX?" in prompt
        assert "Information not found" in prompt

    def test_build_prompt_no_chunks(self):
        """Test prompt with no context."""
        builder = PromptBuilder()
        prompt = builder.build_prompt("Test question?", [])
        assert "No relevant context" in prompt

    def test_prompt_stats(self):
        """Test prompt statistics."""
        builder = PromptBuilder()
        chunks = [{"text": "Test content.", "source": "test.txt", "score": 0.9, "chunk_id": 1}]
        prompt = builder.build_prompt("Question?", chunks)
        stats = builder.get_prompt_stats(prompt)
        assert stats["characters"] > 0
        assert stats["words"] > 0
        assert stats["lines"] > 0

# =============================================================================
# Test 8: End-to-End Pipeline
# =============================================================================

class TestRAGPipeline:
    """Test suite for the full RAG pipeline."""

    def test_index_documents(self, sample_documents_dir, tmpdir):
        """Test full indexing pipeline."""
        vector_dir = os.path.join(tmpdir, "vector_store")

        pipeline = RAGPipeline(
            documents_dir=sample_documents_dir,
            vector_store_dir=vector_dir,
        )

        stats = pipeline.index_documents()

        # Verify indexing results
        assert stats["documents_loaded"] >= 1
        assert stats["chunks_created"] >= 1
        assert stats["embeddings_generated"] >= 1
        assert stats["index_vector_count"] >= 1
        assert stats["index_saved"] == True
        assert stats["elapsed_time_seconds"] > 0

    def test_index_and_query(self, sample_documents_dir, tmpdir):
        """Test indexing and querying."""
        vector_dir = os.path.join(tmpdir, "vector_store")

        pipeline = RAGPipeline(
            documents_dir=sample_documents_dir,
            vector_store_dir=vector_dir,
        )

        # Index
        pipeline.index_documents()

        # Query (will not use LLM since it may not be available)
        result = pipeline.query(
            "What is CARIVIX AI?",
            verbose=False,
        )

        assert result["question"] == "What is CARIVIX AI?"
        assert len(result["retrieved_chunks"]) >= 1
        assert result["retrieval_time"] > 0

    def test_save_and_load_index(self, sample_documents_dir, tmpdir):
        """Test saving and loading index across pipeline instances."""
        vector_dir = os.path.join(tmpdir, "vector_store")

        # Create and save index
        pipeline1 = RAGPipeline(
            documents_dir=sample_documents_dir,
            vector_store_dir=vector_dir,
        )
        pipeline1.index_documents()

        # Create new pipeline and load index
        pipeline2 = RAGPipeline(
            documents_dir=sample_documents_dir,
            vector_store_dir=vector_dir,
        )
        loaded = pipeline2.load_index()
        assert loaded == True

        # Query using loaded index
        result = pipeline2.query("CARIVIX AI", verbose=False)
        assert len(result["retrieved_chunks"]) >= 1

# =============================================================================
# Run tests directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


