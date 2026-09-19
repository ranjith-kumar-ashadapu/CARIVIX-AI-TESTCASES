"""
Response Generator Module for CARIVIX AI RAG Pipeline
======================================================

Generates context-aware responses using an LLM.

Supports:
    - Ollama (Llama 3.1, Mistral, etc.) via REST API
    - HuggingFace pipeline (local models)

The interface is modular so additional LLM backends can be swapped
in by implementing the BaseLLM abstract interface.

Usage:
    # Ollama
    generator = ResponseGenerator(
        backend="ollama",
        model_name="llama3.1",
        base_url="http://localhost:11434",
    )

    # HuggingFace
    generator = ResponseGenerator(
        backend="huggingface",
        model_name="microsoft/phi-2",
    )

    response = generator.generate(prompt)
"""

import os
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generator

from rag.config import DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_BASE_URL
from src.utils import ensure_directory

logger = logging.getLogger("CARIVIX_AI")

# =============================================================================
# Abstract Base Class for LLM Backends
# =============================================================================

class BaseLLM(ABC):
    """
    Abstract base class for LLM backends.

    Implement this interface to add support for new LLM providers.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: Input prompt string.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            Generated text response.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM backend is available and reachable.

        Returns:
            True if the backend is ready for inference.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name identifier."""
        ...

# =============================================================================
# Ollama Backend
# =============================================================================

class OllamaLLM(BaseLLM):
    """
    LLM backend using Ollama's REST API.

    Connects to a local Ollama server running at the specified URL.
    Supports any model available in Ollama (llama3.1, mistral, etc.).

    Attributes:
        model_name: Name of the Ollama model to use.
        base_url: URL of the Ollama server.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: int = 300,
    ) -> None:
        """
        Initialize the Ollama LLM backend.

        Args:
            model_name: Ollama model name (e.g., 'llama3.1', 'mistral').
            base_url: Ollama server URL.
            timeout: Request timeout in seconds.
        """
        self._model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        logger.info(
            "OllamaLLM initialized. Model: %s, URL: %s",
            model_name,
            base_url,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Generate a response using the Ollama API.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text response.

        Raises:
            ConnectionError: If the Ollama server is unreachable.
            RuntimeError: If the API returns an error.
        """
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests is required for Ollama. Install: pip install requests"
            )

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        # Merge any additional kwargs
        payload["options"].update(kwargs)

        logger.debug(
            "Ollama request: model=%s, max_tokens=%d, temperature=%.2f",
            self._model_name,
            max_tokens,
            temperature,
        )

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            generated_text = result.get("response", "").strip()

            logger.debug(
                "Ollama response generated. Length: %d chars",
                len(generated_text),
            )
            return generated_text

        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? (ollama serve)"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Ollama API error: {exc}"
            ) from exc

    def is_available(self) -> bool:
        """
        Check if the Ollama server is reachable.

        Returns:
            True if the server responds to a health check.
        """
        try:
            import requests
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    @property
    def model_name(self) -> str:
        return self._model_name

# =============================================================================
# HuggingFace Backend
# =============================================================================

class HuggingFaceLLM(BaseLLM):
    """
    LLM backend using HuggingFace Transformers pipeline.

    Runs models locally using the transformers library.
    Suitable for smaller models that can run on CPU/GPU.

    Attributes:
        model_name: HuggingFace model identifier.
        pipeline: The loaded transformers pipeline.
        device: Device to run inference on.
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        device: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> None:
        """
        Initialize the HuggingFace LLM backend.

        Args:
            model_name: HuggingFace model ID (e.g., 'microsoft/phi-2',
                'mistralai/Mistral-7B-Instruct-v0.2').
            device: Device ('cpu', 'cuda', 'auto'). If None, auto-detects.
            max_new_tokens: Default max tokens for generation.
        """
        self._model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

        # Auto-detect device
        if device is None:
            try:
                import torch
                self.device = 0 if torch.cuda.is_available() else -1
                self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = -1
                self.device_str = "cpu"
        elif device == "cpu":
            self.device = -1
            self.device_str = "cpu"
        else:
            self.device = 0
            self.device_str = device

        logger.info(
            "HuggingFaceLLM initialized. Model: %s, Device: %s",
            model_name,
            self.device_str,
        )

    def _load_pipeline(self):
        """
        Lazy-load the HuggingFace text generation pipeline.
        """
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required for HuggingFace backend. "
                "Install: pip install transformers"
            ) from exc

        logger.info(
            "Loading HuggingFace pipeline for '%s' (device: %s)...",
            self._model_name,
            self.device_str,
        )

        self._pipeline = hf_pipeline(
            "text-generation",
            model=self._model_name,
            device=self.device,
        )

        logger.info("HuggingFace pipeline loaded.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Generate a response using the HuggingFace pipeline.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text response.
        """
        self._load_pipeline()

        logger.debug(
            "HF generation: max_tokens=%d, temperature=%.2f",
            max_tokens,
            temperature,
        )

        result = self._pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self._pipeline.tokenizer.eos_token_id,
            **kwargs,
        )

        generated_text = result[0]["generated_text"].strip()

        # Remove the input prompt from the output (for text-generation pipelines)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        logger.debug(
            "HF response generated. Length: %d chars",
            len(generated_text),
        )
        return generated_text

    def is_available(self) -> bool:
        """
        Check if the HuggingFace model can be loaded.

        Returns:
            True if the model is available.
        """
        try:
            self._load_pipeline()
            return True
        except Exception:
            return False

    @property
    def model_name(self) -> str:
        return self._model_name

# =============================================================================
# Response Generator (Facade)
# =============================================================================

class ResponseGenerator:
    """
    Facade for generating responses using different LLM backends.

    Provides a unified interface regardless of the underlying LLM
    provider (Ollama, HuggingFace, etc.).

    Usage:
        generator = ResponseGenerator(
            backend="ollama",
            model_name="llama3.1",
        )
        response = generator.generate("What is CARIVIX AI?")
    """

    SUPPORTED_BACKENDS = {
        "ollama": OllamaLLM,
        "huggingface": HuggingFaceLLM,
    }

    def __init__(
        self,
        backend: str = "ollama",
        model_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the ResponseGenerator.

        Args:
            backend: LLM backend type ('ollama' or 'huggingface').
            model_name: Model name for the backend.
                Defaults: Ollama → 'llama3.1', HuggingFace → 'microsoft/phi-2'.
            **kwargs: Additional backend-specific arguments.

        Raises:
            ValueError: If the backend is not supported.
        """
        backend = backend.lower()
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                f"Supported: {list(self.SUPPORTED_BACKENDS.keys())}"
            )

        # Set default model names
        if model_name is None:
            if backend == "ollama":
                model_name = DEFAULT_LLM_MODEL
            elif backend == "huggingface":
                model_name = "microsoft/phi-2"

        backend_class = self.SUPPORTED_BACKENDS[backend]
        self.llm = backend_class(model_name=model_name, **kwargs)

        logger.info(
            "ResponseGenerator initialized. Backend: %s, Model: %s",
            backend,
            self.llm.model_name,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: Input prompt string.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.

        Returns:
            Generated text response.
        """
        logger.info(
            "Generating response (max_tokens=%d, temperature=%.2f)...",
            max_tokens,
            temperature,
        )

        start_time = time.time()
        response = self.llm.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        elapsed = time.time() - start_time

        logger.info(
            "Response generated in %.2f seconds. Length: %d chars",
            elapsed,
            len(response),
        )

        return response

    def is_available(self) -> bool:
        """
        Check if the LLM backend is available.

        Returns:
            True if the backend is ready.
        """
        return self.llm.is_available()

    @property
    def model_name(self) -> str:
        """
        Return the current model name.
        """
        return self.llm.model_name

    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about the current backend configuration.

        Returns:
            Dictionary with backend details.
        """
        return {
            "backend": type(self.llm).__name__,
            "model_name": self.llm.model_name,
            "available": self.is_available(),
        }

    def generate_grounded_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        prompt_template: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a grounded answer using the provided context."""
        started = time.time()
        if not query or not query.strip():
            return {
                "answer": "I could not answer because the query was empty.",
                "sources": [],
                "retrieved_chunks": retrieved_chunks,
                "model": self.model_name,
                "latency": {"total_latency": 0.0},
            }

        prompt = prompt_template or ""
        if not prompt:
            from rag.prompt_builder import PromptBuilder
            prompt = PromptBuilder().build_prompt(query=query, retrieved_chunks=retrieved_chunks)

        answer = self.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
        latency = {"total_latency": round(time.time() - started, 4)}
        sources = []
        for chunk in retrieved_chunks:
            source = chunk.get("source") or chunk.get("metadata", {}).get("source") or "unknown"
            if source not in sources:
                sources.append(source)

        return {
            "answer": answer.strip(),
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "model": self.model_name,
            "latency": latency,
        }

