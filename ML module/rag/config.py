"""Central configuration for the CARIVIX AI RAG pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.utils import load_config

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_RETRIEVAL_K = 5
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "llama3.1"


@dataclass
class RAGConfig:
    """Runtime configuration for the RAG stack."""

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    retrieval_k: int = DEFAULT_RETRIEVAL_K
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    llm_model: str = DEFAULT_LLM_MODEL
    llm_backend: str = "ollama"
    score_threshold: Optional[float] = None
    include_metadata: bool = True
    metadata_fields: Dict[str, Any] = field(
        default_factory=lambda: {
            "document_id": True,
            "file_name": True,
            "source": True,
            "document_type": True,
            "chunk_id": True,
            "chunk_index": True,
        }
    )

    @classmethod
    def from_file(cls, config_path: Optional[str] = None) -> "RAGConfig":
        """Load a YAML config file if present; otherwise return defaults."""
        if not config_path:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            config_path = os.path.join(project_root, "config", "config.yaml")

        if not os.path.exists(config_path):
            return cls()

        loaded = load_config(config_path, base_dir=os.path.dirname(os.path.dirname(__file__)))
        rag_cfg = loaded.get("rag", {}) if isinstance(loaded, dict) else {}
        if not isinstance(rag_cfg, dict):
            rag_cfg = {}

        return cls(
            embedding_model=rag_cfg.get("embedding_model", cls().embedding_model),
            embedding_dimension=int(rag_cfg.get("embedding_dimension", cls().embedding_dimension)),
            chunk_size=int(rag_cfg.get("chunk_size", cls().chunk_size)),
            chunk_overlap=int(rag_cfg.get("chunk_overlap", cls().chunk_overlap)),
            retrieval_k=int(rag_cfg.get("retrieval_k", cls().retrieval_k)),
            ollama_base_url=rag_cfg.get("ollama_base_url", cls().ollama_base_url),
            llm_model=rag_cfg.get("llm_model", cls().llm_model),
            llm_backend=rag_cfg.get("llm_backend", cls().llm_backend),
            score_threshold=rag_cfg.get("score_threshold"),
            include_metadata=bool(rag_cfg.get("include_metadata", cls().include_metadata)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_k": self.retrieval_k,
            "ollama_base_url": self.ollama_base_url,
            "llm_model": self.llm_model,
            "llm_backend": self.llm_backend,
            "score_threshold": self.score_threshold,
            "include_metadata": self.include_metadata,
        }


DEFAULT_RAG_CONFIG = RAGConfig()


def get_rag_config() -> RAGConfig:
    """Return the runtime RAG configuration."""
    return DEFAULT_RAG_CONFIG
