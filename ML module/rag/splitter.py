"""
Text Preprocessing & Splitting Module for CARIVIX AI RAG Pipeline
==================================================================

Handles:
    - Text cleaning (extra spaces, special characters, normalization)
    - Semantic chunking using LangChain's RecursiveCharacterTextSplitter

Configuration:
    - Chunk Size:      500 characters
    - Chunk Overlap:    50 characters
    - Separators:      ["\n\n", "\n", ".", " ", ""] (Recursive default)

Usage:
    preprocessor = TextPreprocessor()
    splitter = DocumentSplitter(chunk_size=500, chunk_overlap=50)

    cleaned_docs = preprocessor.clean_documents(raw_documents)
    chunks = splitter.split_documents(cleaned_docs)
"""

import re
import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("CARIVIX_AI")


def _ensure_document_metadata(document: Document) -> Dict[str, Any]:
    metadata = dict(document.metadata or {})
    metadata.setdefault("document_id", metadata.get("document_id") or metadata.get("source") or "doc")
    metadata.setdefault("file_name", metadata.get("file_name") or metadata.get("source") or "unknown")
    metadata.setdefault("source", metadata.get("source") or metadata.get("file_name") or "unknown")
    metadata.setdefault("document_type", metadata.get("document_type") or "text")
    return metadata


class TextPreprocessor:
    """
    Cleans and normalizes document text while preserving structure.

    Operations:
        - Removes extra whitespace (leading/trailing/multiple spaces)
        - Removes unnecessary special characters (optional)
        - Normalizes line breaks
        - Strips control characters
    """

    def __init__(self, remove_special_chars: bool = False) -> None:
        """
        Initialize the TextPreprocessor.

        Args:
            remove_special_chars: If True, removes non-alphanumeric characters
                except spaces, periods, commas, and newlines.
                Default: False (preserves document structure).
        """
        self.remove_special_chars = remove_special_chars
        logger.info(
            "TextPreprocessor initialized. Remove special chars: %s",
            remove_special_chars,
        )

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.

        Args:
            text: Raw text to clean.

        Returns:
            Cleaned text string.
        """
        if not text or not isinstance(text, str):
            return ""

        # Strip leading/trailing whitespace
        text = text.strip()

        # Normalize line breaks: \r\n → \n, \r → \n
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove null bytes and other control characters (except newlines/tabs)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Collapse multiple blank lines into one
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces (but preserve single spaces)
        text = re.sub(r" +", " ", text)

        # Collapse multiple tabs
        text = re.sub(r"\t+", "\t", text)

        # Remove spaces at the beginning of lines
        text = re.sub(r"\n +", "\n", text)

        # Remove spaces before newlines
        text = re.sub(r" +\n", "\n", text)

        if self.remove_special_chars:
            # Keep only alphanumeric, spaces, periods, commas, newlines
            text = re.sub(r"[^\w\s.,;:!?'\"()-]", "", text)
            text = re.sub(r"\s+", " ", text)

        return text.strip()

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        Clean a list of LangChain Document objects in place.

        Args:
            documents: List of Document objects to clean.

        Returns:
            List of cleaned Document objects (same length as input).
        """
        cleaned: List[Document] = []
        for doc in documents:
            cleaned_text = self.clean_text(doc.page_content)
            if cleaned_text:  # Skip documents that become empty
                metadata = _ensure_document_metadata(doc)
                cleaned.append(
                    Document(
                        page_content=cleaned_text,
                        metadata=metadata,
                    )
                )
            else:
                logger.debug(
                    "Document became empty after cleaning: %s",
                    doc.metadata.get("source", "unknown"),
                )

        logger.info(
            "Text cleaning complete. %d → %d documents.",
            len(documents),
            len(cleaned),
        )
        return cleaned

    @staticmethod
    def word_count(text: str) -> int:
        """
        Count the number of words in a text string.

        Args:
            text: Input text.

        Returns:
            Word count.
        """
        return len(text.split())

    @staticmethod
    def character_count(text: str) -> int:
        """
        Count the number of characters in a text string.

        Args:
            text: Input text (excluding whitespace).

        Returns:
            Character count.
        """
        return len(text.replace(" ", "").replace("\n", ""))

class DocumentSplitter:
    """
    Splits documents into semantic chunks using recursive character splitting.

    Uses RecursiveCharacterTextSplitter which attempts to split on:
        1. Double newlines (paragraphs)
        2. Single newlines
        3. Periods (sentences)
        4. Spaces (words)
        5. Characters (last resort)

    Configuration:
        chunk_size:    Maximum characters per chunk (default: 500)
        chunk_overlap: Overlap between consecutive chunks (default: 50)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        length_function: callable = len,
        separators: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the DocumentSplitter.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
            length_function: Function to measure text length.
            separators: Custom list of separators for splitting.
                Default: ["\\n\\n", "\\n", ".", " ", ""]
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if separators is None:
            separators = ["\n\n", "\n", ".", " ", ""]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=separators,
        )

        logger.info(
            "DocumentSplitter initialized. "
            "Chunk size: %d, Overlap: %d",
            chunk_size,
            chunk_overlap,
        )

    def split_documents(
        self, documents: List[Document]
    ) -> List[Document]:
        """
        Split a list of documents into smaller chunks.

        Each chunk inherits the metadata of its parent document, plus
        a 'chunk_id' and 'chunk_total' to track which chunk of the
        parent document it belongs to.

        Args:
            documents: List of cleaned Document objects to split.

        Returns:
            List of smaller Document chunks with chunk metadata.
        """
        if not documents:
            logger.warning("No documents to split.")
            return []

        chunks = self.text_splitter.split_documents(documents)

        valid_chunks: List[Document] = []
        for chunk in chunks:
            if not chunk.page_content or not chunk.page_content.strip():
                continue
            metadata = _ensure_document_metadata(chunk)
            metadata.setdefault("document_id", metadata.get("document_id") or metadata.get("source") or "doc")
            metadata.setdefault("file_name", metadata.get("file_name") or metadata.get("source") or "unknown")
            metadata.setdefault("source", metadata.get("source") or metadata.get("file_name") or "unknown")
            metadata.setdefault("document_type", metadata.get("document_type") or "text")
            chunk.metadata = metadata
            valid_chunks.append(chunk)

        # Add chunk tracking metadata
        doc_chunk_map: dict = {}
        for chunk in valid_chunks:
            document_id = chunk.metadata.get("document_id", "unknown")
            doc_chunk_map.setdefault(document_id, []).append(chunk)

        for document_id, source_chunks in doc_chunk_map.items():
            total = len(source_chunks)
            for idx, chunk in enumerate(source_chunks, start=1):
                chunk.metadata["chunk_id"] = f"{document_id}-chunk-{idx}"
                chunk.metadata["chunk_index"] = idx - 1
                chunk.metadata["chunk_total"] = total
                chunk.metadata["document_id"] = document_id

        logger.info(
            "Document splitting complete. %d documents → %d chunks. "
            "Chunk size: %d, Overlap: %d",
            len(documents),
            len(valid_chunks),
            self.chunk_size,
            self.chunk_overlap,
        )

        if valid_chunks:
            chunk_lengths = [len(c.page_content) for c in valid_chunks]
            logger.debug(
                "Chunk stats: min=%d, max=%d, avg=%.1f",
                min(chunk_lengths),
                max(chunk_lengths),
                sum(chunk_lengths) / len(chunk_lengths),
            )

        return valid_chunks

    def split_text(self, text: str, source: str = "unknown") -> List[Document]:
        """
        Split a single text string into chunks.

        Useful for testing or processing individual texts without
        full Document objects.

        Args:
            text: Text content to split.
            source: Source name for metadata.

        Returns:
            List of Document chunks.
        """
        doc = Document(
            page_content=text,
            metadata={
                "source": source,
                "file_name": source,
                "document_type": "text",
                "document_id": f"text-{source}",
            },
        )
        return self.split_documents([doc])

    @property
    def config(self) -> dict:
        """
        Return the current splitter configuration.

        Returns:
            Dictionary with chunk_size and chunk_overlap.
        """
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


