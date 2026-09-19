"""
Vector Database Module for CARIVIX AI RAG Pipeline
====================================================

Manages the FAISS vector index for storing and searching document
embeddings.

Capabilities:
    - Create a FAISS index from document chunks and embeddings
    - Save the index and document metadata to disk
    - Load a previously saved index
    - Search for nearest neighbors given a query embedding

Storage Layout:
    {index_directory}/
        index.faiss     - FAISS index file
        index.pkl       - Document metadata (pickled list of Document objects)

Usage:
    store = VectorStore(index_directory="data/vector_store/")
    store.create_index(chunks, embeddings)
    store.save_index()
    store.load_index()
    results = store.similarity_search(query_vector, k=5)
"""

import os
import pickle
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import faiss
from langchain_core.documents import Document

from src.utils import ensure_directory

logger = logging.getLogger("CARIVIX_AI")

class VectorStore:
    """
    FAISS-based vector store for document embeddings.

    Stores embeddings in a FAISS index (L2 distance by default) and
    maintains a parallel list of Document objects for metadata retrieval.

    Attributes:
        index_directory: Directory path for saving/loading the index.
        index: FAISS index object (created after create_index or load_index).
        documents: List of Document objects aligned with index vectors.
        dimension: Embedding vector dimension.
        index_type: Type of FAISS index ('L2' or 'IP' for inner product).
    """

    def __init__(
        self,
        index_directory: str = "data/vector_store/",
        index_type: str = "L2",
    ) -> None:
        """
        Initialize the VectorStore.

        Args:
            index_directory: Directory to persist the FAISS index and metadata.
            index_type: FAISS index type ('L2' for Euclidean, 'IP' for inner product).
        """
        self.index_directory = index_directory
        self.index_type = index_type.upper()

        if self.index_type not in ("L2", "IP"):
            logger.warning(
                "Unsupported index type '%s'. Falling back to 'L2'.",
                index_type,
            )
            self.index_type = "L2"

        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []
        self.dimension: Optional[int] = None

        logger.info(
            "VectorStore initialized. Directory: %s, Index type: %s",
            os.path.abspath(self.index_directory),
            self.index_type,
        )

    # ------------------------------------------------------------------
    # Index Creation
    # ------------------------------------------------------------------

    def create_index(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
    ) -> None:
        """
        Create a new FAISS index from documents and their embeddings.

        Args:
            documents: List of Document objects (must match embeddings count).
            embeddings: numpy array of shape (n_docs, embedding_dim).

        Raises:
            ValueError: If document count doesn't match embedding count
                or if embeddings array is empty.
        """
        if len(documents) == 0:
            raise ValueError("Cannot create index with empty documents list.")

        if embeddings.shape[0] == 0:
            raise ValueError("Cannot create index with empty embeddings.")

        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"Document count ({len(documents)}) must match "
                f"embedding count ({embeddings.shape[0]})."
            )

        self.dimension = embeddings.shape[1]
        self.documents = documents.copy()

        # Build FAISS index based on dimension
        if self.index_type == "IP":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

        # Add embeddings to the index
        embeddings_np = np.ascontiguousarray(
            embeddings.astype(np.float32)
        )
        self.index.add(embeddings_np)

        logger.info(
            "FAISS index created. Type: %s, Dimension: %d, Vectors: %d",
            self.index_type,
            self.dimension,
            self.index.ntotal,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, directory: Optional[str] = None) -> Tuple[str, str]:
        """
        Save the FAISS index and document metadata to disk.

        Args:
            directory: Override directory. Uses self.index_directory if None.

        Returns:
            Tuple of (index_file_path, metadata_file_path).

        Raises:
            ValueError: If no index has been created or loaded.
        """
        if self.index is None:
            raise ValueError(
                "No index to save. Call create_index() or load_index() first."
            )

        save_dir = directory or self.index_directory
        ensure_directory(save_dir)

        # Save FAISS index
        index_path = os.path.join(save_dir, "index.faiss")
        faiss.write_index(self.index, index_path)

        # Save document metadata
        metadata_path = os.path.join(save_dir, "index.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "dimension": self.dimension,
                    "index_type": self.index_type,
                    "vector_count": self.index.ntotal,
                },
                f,
            )

        logger.info(
            "Index saved. FAISS: %s, Metadata: %s "
            "(vectors: %d, dimension: %d)",
            index_path,
            metadata_path,
            self.index.ntotal,
            self.dimension,
        )

        return index_path, metadata_path

    def load_index(self, directory: Optional[str] = None) -> None:
        """
        Load a previously saved FAISS index and document metadata.

        Args:
            directory: Override directory. Uses self.index_directory if None.

        Raises:
            FileNotFoundError: If index files don't exist.
        """
        load_dir = directory or self.index_directory
        index_path = os.path.join(load_dir, "index.faiss")
        metadata_path = os.path.join(load_dir, "index.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index file not found: {index_path}"
            )
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        try:
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            self.documents = metadata.get("documents", [])
            self.dimension = metadata.get("dimension")
            if self.dimension is None and self.documents:
                self.dimension = len(self.documents[0].page_content)
        except (FileNotFoundError, OSError, pickle.PickleError, EOFError, ValueError) as exc:
            logger.warning("Unable to load FAISS index from %s: %s", load_dir, exc)
            self.clear()
            raise

        logger.info(
            "Index loaded. Vectors: %d, Dimension: %d, Type: %s",
            self.index.ntotal,
            self.dimension,
            self.index_type,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search the index for the k nearest neighbors to a query vector.

        Args:
            query_vector: Query embedding vector of shape (dimension,) or (1, dimension).
            k: Number of nearest neighbors to return.

        Returns:
            List of dictionaries, each containing:
                - 'score':     Similarity/distance score
                - 'document':  The matching Document object
                - 'chunk_id':  Chunk identifier (from document metadata)
                - 'source':    Source document name

        Raises:
            ValueError: If no index is loaded/created.
        """
        if self.index is None:
            raise ValueError(
                "No index available. Call create_index() or load_index() first."
            )

        if query_vector is None or np.size(query_vector) == 0:
            logger.warning("Empty query vector provided for similarity search.")
            return []

        if self.index is None:
            raise ValueError(
                "No index available. Call create_index() or load_index() first."
            )

        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty.")
            return []

        # Ensure query vector is 2D float32
        q = np.asarray(query_vector)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if self.dimension is not None and q.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dimension}, got {q.shape[1]}."
            )
        q = np.ascontiguousarray(q.astype(np.float32))

        top_k = min(int(k), self.index.ntotal)
        if top_k <= 0:
            return []

        distances, indices = self.index.search(q, top_k)

        results: List[Dict[str, Any]] = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1 or idx >= len(self.documents):
                continue

            doc = self.documents[idx]
            metadata = dict(doc.metadata or {})
            if self.index_type == "L2":
                score = float(1.0 / (1.0 + dist))
            else:
                score = float(dist)

            results.append(
                {
                    "rank": rank + 1,
                    "score": round(score, 6),
                    "distance": float(dist),
                    "document": doc,
                    "document_id": metadata.get("document_id"),
                    "file_name": metadata.get("file_name") or metadata.get("source", "unknown"),
                    "source": metadata.get("source", "unknown"),
                    "document_type": metadata.get("document_type", "text"),
                    "chunk_id": metadata.get("chunk_id", idx),
                    "chunk_index": metadata.get("chunk_index", idx),
                    "page": metadata.get("page", None),
                    "metadata": metadata,
                    "text": doc.page_content,
                }
            )

        logger.debug(
            "Similarity search returned %d results for query.",
            len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Index Information
    # ------------------------------------------------------------------

    @property
    def vector_count(self) -> int:
        """
        Return the number of vectors currently in the index.

        Returns:
            Integer count, or 0 if no index is loaded.
        """
        if self.index is None:
            return 0
        return self.index.ntotal

    @property
    def is_index_loaded(self) -> bool:
        """
        Check whether an index is currently loaded.

        Returns:
            True if an index is available.
        """
        return self.index is not None

    def get_index_info(self) -> Dict[str, Any]:
        """
        Return metadata about the current index state.

        Returns:
            Dictionary with index statistics.
        """
        return {
            "vector_count": self.vector_count,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "document_count": len(self.documents),
            "index_loaded": self.is_index_loaded,
            "index_directory": os.path.abspath(self.index_directory),
        }

    def clear(self) -> None:
        """
        Clear the current index and document list.
        """
        self.index = None
        self.documents = []
        self.dimension = None
        logger.info("Vector store cleared.")


