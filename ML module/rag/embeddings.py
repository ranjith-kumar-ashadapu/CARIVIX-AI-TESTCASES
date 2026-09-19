"""
Embedding Generation Module for CARIVIX AI RAG Pipeline
========================================================

Generates dense vector embeddings for text chunks using
Sentence Transformers ('all-MiniLM-L6-v2').

Features:
    - Batch processing for efficiency
    - Optional model caching to disk
    - Configurable device (CPU / CUDA)
    - Embedding dimension reporting

Usage:
    generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    embeddings = generator.generate(["text chunk 1", "text chunk 2"])
    vector = generator.generate_single("single query text")
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from langchain_core.documents import Document

from rag.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_DIMENSION

logger = logging.getLogger("CARIVIX_AI")

class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using Sentence Transformers.

    Wraps the 'sentence-transformers' library with the
    'all-MiniLM-L6-v2' model by default, providing batched embedding
    generation and utility methods.

    Attributes:
        model_name: Name of the Sentence Transformer model.
        model: The loaded SentenceTransformer model instance.
        dimension: Embedding vector dimension (384 for all-MiniLM-L6-v2).
        device: Device used for inference ('cpu' or 'cuda').
        batch_size: Number of texts to process per batch.
    """

    DEFAULT_MODEL = DEFAULT_EMBEDDING_MODEL
    EXPECTED_DIMENSION = DEFAULT_EMBEDDING_DIMENSION

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32,
        cache_folder: Optional[str] = None,
    ) -> None:
        """
        Initialize the EmbeddingGenerator.

        Args:
            model_name: Name or path of the Sentence Transformer model.
                Default: 'all-MiniLM-L6-v2'.
            device: Device for inference ('cpu', 'cuda').
                If None, auto-detects CUDA availability.
            batch_size: Number of texts to process simultaneously.
            cache_folder: Custom cache directory for model weights.
                If None, uses default HuggingFace cache.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None  # Lazy-loaded

        # Auto-detect device
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self.cache_folder = cache_folder
        self.dimension: int = self.EXPECTED_DIMENSION

        logger.info(
            "EmbeddingGenerator initialized. Model: %s, Device: %s, Batch: %d",
            model_name,
            self.device,
            batch_size,
        )

    # ------------------------------------------------------------------
    # Model Lazy Loading
    # ------------------------------------------------------------------

    @property
    def model(self):
        """
        Lazy-load the SentenceTransformer model on first access.
        """
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        """
        Load the Sentence Transformer model from HuggingFace Hub.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install: pip install sentence-transformers"
            ) from exc

        logger.info(
            "Loading SentenceTransformer model: %s (device: %s)...",
            self.model_name,
            self.device,
        )

        model_kwargs = {"device": self.device}
        if self.cache_folder:
            model_kwargs["cache_folder"] = self.cache_folder

        self._model = SentenceTransformer(
            self.model_name, **model_kwargs
        )

        # Confirm dimension
        test_embedding = self._model.encode(
            "Test sentence.", convert_to_numpy=True
        )
        self.dimension = test_embedding.shape[0]
        logger.info(
            "Model loaded. Embedding dimension: %d", self.dimension
        )

    # ------------------------------------------------------------------
    # Embedding Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of text strings.

        Args:
            texts: List of text strings to embed.
            show_progress: Whether to display a progress bar.

        Returns:
            numpy array of shape (len(texts), embedding_dimension).

        Raises:
            ValueError: If texts list is empty.
        """
        if not texts:
            raise ValueError("Cannot generate embeddings for empty text list.")

        cleaned_texts = []
        for text in texts:
            if not isinstance(text, str):
                continue
            cleaned = text.strip()
            if cleaned:
                cleaned_texts.append(cleaned)

        if not cleaned_texts:
            raise ValueError("Cannot generate embeddings for empty text list after filtering.")

        logger.debug(
            "Generating embeddings for %d texts (batch size: %d)...",
            len(cleaned_texts),
            self.batch_size,
        )

        try:
            embeddings = self.model.encode(
                cleaned_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        except Exception as exc:  # pragma: no cover - defensive error handling
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc

        embeddings = np.asarray(embeddings)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[1] != self.dimension:
            logger.warning(
                "Embedding dimension mismatch detected: expected %d, got %d",
                self.dimension,
                embeddings.shape[1],
            )
        logger.debug(
            "Embeddings generated. Shape: %s", embeddings.shape
        )
        return embeddings

    def generate_single(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a single text string.

        Args:
            text: Single text string to embed.

        Returns:
            numpy array of shape (embedding_dimension,).

        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text.")

        try:
            embedding = self.model.encode(
                text.strip(), convert_to_numpy=True
            )
        except Exception as exc:  # pragma: no cover - defensive error handling
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc

        embedding = np.asarray(embedding)
        if embedding.ndim == 0:
            embedding = embedding.reshape(1)
        if embedding.ndim == 2:
            embedding = embedding.reshape(-1)
        return embedding

    def generate_from_documents(
        self,
        documents: List[Document],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings from a list of LangChain Documents.

        Extracts page_content from each Document and generates
        embeddings.

        Args:
            documents: List of Document objects.
            show_progress: Whether to display a progress bar.

        Returns:
            numpy array of shape (len(documents), embedding_dimension).
        """
        texts = [doc.page_content for doc in documents]
        return self.generate(texts, show_progress=show_progress)

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate an embedding optimized for retrieval queries.

        For Sentence Transformers with symmetric models like
        all-MiniLM-L6-v2, this is the same as generate_single.

        Args:
            query: User query string.

        Returns:
            numpy array of shape (embedding_dimension,).
        """
        return self.generate_single(query)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_dimension(self) -> int:
        """
        Return the embedding dimension of the loaded model.

        Returns:
            Integer dimension (384 for all-MiniLM-L6-v2).
        """
        # Trigger model load if not already loaded
        _ = self.model
        return self.dimension

    def get_model_info(self) -> dict:
        """
        Return information about the loaded embedding model.

        Returns:
            Dictionary with model metadata.
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_seq_length": getattr(
                self.model, "max_seq_length", "unknown"
            )
            if self._model
            else "not loaded",
        }


