"""
End-to-End RAG Pipeline Module for CARIVIX AI
===============================================

Orchestrates the complete RAG pipeline:

    Load Documents
        ↓
    Preprocess & Split into Chunks
        ↓
    Generate Embeddings
        ↓
    Create FAISS Index
        ↓
    Save Index
        ↓
    User Query → Retrieve Top-5 Chunks → Build Prompt → LLM → Response

Usage:
    pipeline = RAGPipeline(
        documents_dir="data/documents/",
        vector_store_dir="data/vector_store/",
        llm_backend="ollama",
        llm_model="llama3.1",
    )

    # Index documents
    pipeline.index_documents()

    # Query
    response = pipeline.query("What is CARIVIX AI?")
"""

import os
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from src.utils import ensure_directory, get_timestamp
from rag.config import DEFAULT_RAG_CONFIG, RAGConfig
from rag.context_builder import ContextBuilder
from rag.loader import DocumentLoader
from rag.splitter import TextPreprocessor, DocumentSplitter
from rag.embeddings import EmbeddingGenerator
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.prompt_builder import PromptBuilder
from rag.generator import ResponseGenerator

logger = logging.getLogger("CARIVIX_AI")

class RAGPipeline:
    """
    End-to-end RAG pipeline orchestrator.

    Combines document loading, splitting, embedding, indexing,
    retrieval, prompt building, and response generation into a
    single pipeline with configurable components.

    Attributes:
        documents_dir: Directory containing source documents.
        vector_store_dir: Directory for FAISS index persistence.
        loader: DocumentLoader instance.
        preprocessor: TextPreprocessor instance.
        splitter: DocumentSplitter instance.
        embedding_generator: EmbeddingGenerator instance.
        vector_store: VectorStore instance.
        retriever: Retriever instance.
        prompt_builder: PromptBuilder instance.
        generator: ResponseGenerator instance.
    """

    def __init__(
        self,
        documents_dir: str = "data/documents/",
        vector_store_dir: str = "data/vector_store/",
        # Embedding config
        embedding_model: Optional[str] = None,
        embedding_device: Optional[str] = None,
        # Chunking config
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        # LLM config
        llm_backend: str = "ollama",
        llm_model: Optional[str] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        # Prompt config
        system_prompt: Optional[str] = None,
        include_metadata: Optional[bool] = None,
        # Retriever config
        retrieval_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> None:
        """
        Initialize the RAGPipeline with all components.

        Args:
            documents_dir: Directory containing source documents.
            vector_store_dir: Directory for FAISS index persistence.
            embedding_model: Sentence Transformer model name.
            embedding_device: Device for embeddings ('cpu', 'cuda').
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.
            llm_backend: LLM backend ('ollama' or 'huggingface').
            llm_model: Model name for the LLM backend.
            llm_kwargs: Additional kwargs for the LLM backend.
            system_prompt: Custom system prompt for the LLM.
            include_metadata: Include metadata in prompt context.
            retrieval_k: Number of chunks to retrieve per query.
            score_threshold: Minimum similarity score for retrieval.
        """
        config = DEFAULT_RAG_CONFIG
        self.documents_dir = documents_dir
        self.vector_store_dir = vector_store_dir
        self.chunk_size = chunk_size if chunk_size is not None else config.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap
        self.retrieval_k = retrieval_k if retrieval_k is not None else config.retrieval_k
        self.score_threshold = score_threshold if score_threshold is not None else config.score_threshold

        # Initialize components
        self.loader = DocumentLoader(documents_dir=documents_dir)
        self.preprocessor = TextPreprocessor()
        self.splitter = DocumentSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.embedding_generator = EmbeddingGenerator(
            model_name=embedding_model or config.embedding_model,
            device=embedding_device,
        )
        self.vector_store = VectorStore(
            index_directory=vector_store_dir,
        )
        self.retriever = Retriever(
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store,
        )
        self.prompt_builder = PromptBuilder(
            system_prompt=system_prompt,
            include_metadata=include_metadata if include_metadata is not None else config.include_metadata,
        )
        self.context_builder = ContextBuilder(max_context_chars=4000)

        # LLM (lazy initialization)
        self._llm_backend = llm_backend
        self._llm_model = llm_model or config.llm_model
        self._llm_kwargs = dict(llm_kwargs or {})
        if self._llm_backend == "ollama" and "base_url" not in self._llm_kwargs:
            self._llm_kwargs["base_url"] = config.ollama_base_url
        self._generator: Optional[ResponseGenerator] = None

        # Pipeline state
        self._is_indexed = False
        self._indexing_stats: Dict[str, Any] = {}

        logger.info(
            "RAGPipeline initialized. Documents: %s, "
            "Vector store: %s, Chunk size: %d/%d, "
            "LLM: %s/%s, Retrieval k: %d",
            documents_dir,
            vector_store_dir,
            chunk_size,
            chunk_overlap,
            llm_backend,
            llm_model or "default",
            retrieval_k,
        )

    # ------------------------------------------------------------------
    # Generator (lazy-loaded)
    # ------------------------------------------------------------------

    @property
    def generator(self) -> ResponseGenerator:
        """Lazy-loaded ResponseGenerator."""
        if self._generator is None:
            self._generator = ResponseGenerator(
                backend=self._llm_backend,
                model_name=self._llm_model,
                **self._llm_kwargs,
            )
        return self._generator

    # ------------------------------------------------------------------
    # Indexing Pipeline
    # ------------------------------------------------------------------

    def index_documents(
        self,
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full indexing pipeline: load → clean → split → embed → index → save.

        Args:
            force_reindex: If True, re-index even if existing index is found.

        Returns:
            Dictionary with indexing statistics:
                - documents_loaded: Number of documents loaded
                - chunks_created: Number of chunks after splitting
                - embeddings_generated: Number of embeddings
                - index_vector_count: Number of vectors in FAISS index
                - index_saved: Whether the index was saved
                - elapsed_time: Total indexing time in seconds
        """
        start_time = time.time()

        # Check if already indexed
        if self._is_indexed and not force_reindex:
            logger.info("Documents already indexed. Use force_reindex=True to re-index.")
            return self._indexing_stats

        logger.info("=" * 60)
        logger.info("  RAG INDEXING PIPELINE")
        logger.info("=" * 60)

        # Step 1: Load Documents
        logger.info("Step 1/5: Loading documents...")
        documents = self.loader.load_all()
        if not documents:
            logger.warning("No documents found in '%s'.", self.documents_dir)
            return {"documents_loaded": 0, "error": "No documents found"}

        # Step 2: Preprocess & Split
        logger.info("Step 2/5: Preprocessing and splitting documents...")
        cleaned_docs = self.preprocessor.clean_documents(documents)
        chunks = self.splitter.split_documents(cleaned_docs)
        if not chunks:
            logger.warning("No chunks created from documents.")
            return {"documents_loaded": len(documents), "chunks_created": 0}

        # Step 3: Generate Embeddings
        logger.info("Step 3/5: Generating embeddings for %d chunks...", len(chunks))
        embeddings = self.embedding_generator.generate_from_documents(
            chunks, show_progress=True
        )

        # Step 4: Create FAISS Index
        logger.info("Step 4/5: Creating FAISS index...")
        self.vector_store.create_index(chunks, embeddings)

        # Step 5: Save Index
        logger.info("Step 5/5: Saving index to '%s'...", self.vector_store_dir)
        index_path, metadata_path = self.vector_store.save_index()

        # Update state
        self._is_indexed = True
        elapsed = time.time() - start_time

        # Collect statistics
        self._indexing_stats = {
            "documents_loaded": len(documents),
            "documents_after_cleaning": len(cleaned_docs),
            "chunks_created": len(chunks),
            "embeddings_generated": embeddings.shape[0],
            "embedding_dimension": embeddings.shape[1],
            "index_vector_count": self.vector_store.vector_count,
            "index_path": index_path,
            "metadata_path": metadata_path,
            "index_saved": True,
            "elapsed_time_seconds": round(elapsed, 2),
            "elapsed_time_formatted": f"{elapsed:.2f}s",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_generator.model_name,
        }

        logger.info("=" * 60)
        logger.info("  INDEXING COMPLETE")
        logger.info("  Documents: %d | Chunks: %d | Embeddings: %d | Vectors: %d",
                     len(documents), len(chunks), embeddings.shape[0],
                     self.vector_store.vector_count)
        logger.info("  Time: %.2f seconds", elapsed)
        logger.info("=" * 60)

        return self._indexing_stats

    def load_index(self) -> bool:
        """
        Load a previously saved index from disk.

        Returns:
            True if the index was loaded successfully, False otherwise.
        """
        try:
            self.vector_store.load_index()
            self._is_indexed = True
            logger.info(
                "Index loaded: %d vectors, %d documents.",
                self.vector_store.vector_count,
                len(self.vector_store.documents),
            )
            return True
        except (FileNotFoundError, Exception) as exc:
            logger.warning("Could not load index: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Query Pipeline
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full query pipeline: retrieve → build prompt → generate response.

        Args:
            question: User's question.
            k: Number of chunks to retrieve (overrides default).
            max_tokens: Maximum tokens for LLM response.
            temperature: LLM sampling temperature.
            verbose: If True, logs detailed pipeline information.

        Returns:
            Dictionary with:
                - question: Original question
                - response: Generated answer
                - retrieved_chunks: List of retrieved chunks with scores
                - prompt: The prompt sent to the LLM
                - retrieval_time: Time for retrieval in seconds
                - generation_time: Time for generation in seconds
                - total_time: Total query time in seconds
        """
        if not self._is_indexed:
            if not self.load_index():
                raise RuntimeError("No index available. Run index_documents() first.")

        if k is None:
            k = self.retrieval_k

        total_start = time.time()

        if verbose:
            logger.info("=" * 60)
            logger.info("  RAG QUERY")
            logger.info("  Question: %s", question[:100])
            logger.info("=" * 60)

        retrieval_start = time.time()
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            k=k,
            score_threshold=self.score_threshold,
        )
        retrieval_time = time.time() - retrieval_start

        if not retrieved_chunks:
            logger.warning("No relevant chunks found for the query.")
            return {
                "question": question,
                "response": "Information not found.",
                "answer": "Information not found.",
                "sources": [],
                "retrieved_chunks": [],
                "prompt": "",
                "model": self.generator.model_name,
                "retrieval_time": round(retrieval_time, 4),
                "generation_time": 0.0,
                "context_construction_time": 0.0,
                "total_time": round(time.time() - total_start, 4),
                "latency": {"query_embedding_latency": 0.0, "retrieval_latency": round(retrieval_time, 4), "context_construction_latency": 0.0, "llm_generation_latency": 0.0, "total_latency": round(time.time() - total_start, 4)},
            }

        context_start = time.time()
        context = self.context_builder.build(retrieved_chunks)
        context_construction_time = time.time() - context_start

        prompt = self.prompt_builder.build_prompt(
            query=question,
            retrieved_chunks=retrieved_chunks,
            max_context_length=4000,
        )

        if verbose:
            logger.info("Retrieved %d chunks in %.4f seconds.", len(retrieved_chunks), retrieval_time)

        generation_start = time.time()
        try:
            if self.generator.is_available():
                response = self.generator.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                logger.warning("LLM backend '%s' is not available. Returning retrieved context only.", self._llm_backend)
                response = "[LLM not available. Retrieved context shown above.]"
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            response = f"Error generating response: {exc}\n\nRetrieved context is available above."

        generation_time = time.time() - generation_start
        total_time = time.time() - total_start
        sources = []
        for chunk in retrieved_chunks:
            source = chunk.get("source") or chunk.get("metadata", {}).get("source") or "unknown"
            if source not in sources:
                sources.append(source)

        result = {
            "question": question,
            "response": response,
            "answer": response,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "context": context,
            "prompt": prompt,
            "model": self.generator.model_name,
            "retrieval_time": round(retrieval_time, 4),
            "generation_time": round(generation_time, 4),
            "context_construction_time": round(context_construction_time, 4),
            "total_time": round(total_time, 4),
            "latency": {
                "query_embedding_latency": 0.0,
                "retrieval_latency": round(retrieval_time, 4),
                "context_construction_latency": round(context_construction_time, 4),
                "llm_generation_latency": round(generation_time, 4),
                "total_latency": round(total_time, 4),
            },
        }

        if verbose:
            logger.info("Response generated in %.4f seconds.", generation_time)
            logger.info("Total query time: %.4f seconds.", total_time)
            logger.info("=" * 60)

        return result

    # ------------------------------------------------------------------
    # Pipeline Information
    # ------------------------------------------------------------------

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the pipeline configuration.

        Returns:
            Dictionary with all configuration parameters.
        """
        return {
            "pipeline": {
                "documents_dir": os.path.abspath(self.documents_dir),
                "vector_store_dir": os.path.abspath(self.vector_store_dir),
                "is_indexed": self._is_indexed,
                "indexing_stats": self._indexing_stats,
            },
            "embedding": self.embedding_generator.get_model_info(),
            "chunking": self.splitter.config,
            "retrieval": {
                "k": self.retrieval_k,
                "score_threshold": self.score_threshold,
            },
            "llm": self.generator.get_backend_info(),
            "prompt": {
                "system_prompt": self.prompt_builder.system_prompt[:100],
                "include_metadata": self.prompt_builder.include_metadata,
            },
        }

    def format_query_result(
        self,
        result: Dict[str, Any],
        include_context: bool = True,
    ) -> str:
        """
        Format a query result for human-readable display.

        Args:
            result: Result dictionary from query().
            include_context: Whether to show retrieved chunks.

        Returns:
            Formatted string for console output.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("  RAG QUERY RESULT")
        lines.append("=" * 70)
        lines.append(f"\n  Question: {result['question']}")
        lines.append(f"\n  Response: {result['response']}")

        if include_context and result.get("retrieved_chunks"):
            lines.append("\n" + "-" * 70)
            lines.append("  Retrieved Context:")
            for chunk in result["retrieved_chunks"]:
                score = chunk.get("score", 0)
                source = chunk.get("source", "unknown")
                text = chunk.get("text", "")[:150]
                lines.append(f"\n    [{score:.4f}] {source}")
                lines.append(f"    {text}...")

        lines.append("\n" + "-" * 70)
        lines.append(
            f"  Retrieval: {result.get('retrieval_time', 0):.4f}s | "
            f"Generation: {result.get('generation_time', 0):.4f}s | "
            f"Total: {result.get('total_time', 0):.4f}s"
        )
        lines.append("=" * 70)

        return "\n".join(lines)

