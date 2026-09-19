"""
Document Loader Module for CARIVIX AI RAG Pipeline
====================================================

Loads documents from various file formats into a unified list of
LangChain Document objects for downstream processing.

Supported Formats:
    - PDF  (.pdf)   via PyPDF2 / pypdf
    - DOCX (.docx)  via python-docx
    - TXT  (.txt)   via built-in open()
    - CSV  (.csv)   via pandas (each row as a document)

Usage:
    loader = DocumentLoader(documents_dir="data/documents/")
    documents = loader.load_all()
"""

import os
import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.utils import ensure_directory

logger = logging.getLogger("CARIVIX_AI")


def _build_document_metadata(filepath: str, file_type: str, extra_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create consistent metadata for every document and chunk."""
    file_name = os.path.basename(filepath)
    document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, filepath))
    metadata = {
        "document_id": document_id,
        "file_name": file_name,
        "source": file_name,
        "document_type": file_type,
        "file_path": filepath,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


class DocumentLoader:
    """
    Loads documents from multiple file formats into LangChain Document objects.

    Scans a specified directory for supported file types and parses each
    into a Document with content and metadata (source, page number, etc.).

    Attributes:
        documents_dir: Path to the directory containing source documents.
        supported_extensions: Mapping of file extensions to loader methods.
        encoding: Text encoding for TXT files (default: utf-8).
    """

    def __init__(
        self,
        documents_dir: str = "data/documents/",
        encoding: str = "utf-8",
    ) -> None:
        """
        Initialize the DocumentLoader.

        Args:
            documents_dir: Directory path containing documents to load.
            encoding: Text encoding to use for plain text files.
        """
        self.documents_dir = documents_dir
        self.encoding = encoding
        self.supported_extensions: Dict[str, str] = {
            ".pdf": "pdf",
            ".docx": "docx",
            ".txt": "txt",
            ".csv": "csv",
        }
        logger.info(
            "DocumentLoader initialized. Directory: %s",
            os.path.abspath(self.documents_dir),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> List[Document]:
        """
        Load all supported documents from the configured directory.

        Scans recursively, parses each file based on its extension,
        and returns a flat list of LangChain Document objects.

        Returns:
            List of LangChain Document objects with content and metadata.
        """
        if not os.path.isdir(self.documents_dir):
            logger.warning(
                "Documents directory does not exist: %s. Creating it.",
                self.documents_dir,
            )
            ensure_directory(self.documents_dir)
            return []

        all_documents: List[Document] = []
        file_count = 0

        for root, _, files in os.walk(self.documents_dir):
            for filename in sorted(files):
                filepath = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1].lower()

                if ext in self.supported_extensions:
                    try:
                        documents = self._load_file(filepath, ext)
                        all_documents.extend(documents)
                        file_count += 1
                        logger.info(
                            "Loaded %s → %d document(s)",
                            filename,
                            len(documents),
                        )
                    except Exception as exc:
                        logger.error(
                            "Failed to load '%s': %s", filename, exc
                        )

        logger.info(
            "Document loading complete. %d file(s), %d document(s) loaded.",
            file_count,
            len(all_documents),
        )
        return all_documents

    def load_file(self, filepath: str) -> List[Document]:
        """
        Load a single file (regardless of configured directory).

        Args:
            filepath: Absolute or relative path to the file.

        Returns:
            List of LangChain Document objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is unsupported.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        if ext not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {list(self.supported_extensions.keys())}"
            )

        return self._load_file(filepath, ext)

    # ------------------------------------------------------------------
    # File-specific Loaders
    # ------------------------------------------------------------------

    def _load_file(self, filepath: str, ext: str) -> List[Document]:
        """
        Dispatch to the correct loader based on file extension.

        Args:
            filepath: Path to the file.
            ext: File extension (e.g., '.pdf').

        Returns:
            List of Document objects.
        """
        loader_map = {
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".txt": self._load_txt,
            ".csv": self._load_csv,
        }
        loader = loader_map.get(ext)
        if loader is None:
            return []
        return loader(filepath)

    def _load_pdf(self, filepath: str) -> List[Document]:
        """
        Load a PDF file using pypdf.

        Each page becomes a separate Document with page metadata.

        Args:
            filepath: Path to the PDF file.

        Returns:
            List of Document objects (one per page).
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.error(
                "pypdf is required for PDF loading. Install: pip install pypdf"
            )
            return []

        documents: List[Document] = []
        filename = os.path.basename(filepath)

        try:
            reader = PdfReader(filepath)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(
                        Document(
                            page_content=text.strip(),
                            metadata=_build_document_metadata(
                                filepath,
                                "pdf",
                                {
                                    "page": page_num,
                                    "total_pages": len(reader.pages),
                                    "file_type": "pdf",
                                },
                            ),
                        )
                    )
        except Exception as exc:
            logger.error("Error reading PDF '%s': %s", filename, exc)

        return documents

    def _load_docx(self, filepath: str) -> List[Document]:
        """
        Load a DOCX file using python-docx.

        Args:
            filepath: Path to the DOCX file.

        Returns:
            List containing a single Document with full text content.
        """
        try:
            from docx import Document as DocxDocument
        except ImportError:
            logger.error(
                "python-docx is required for DOCX loading. "
                "Install: pip install python-docx"
            )
            return []

        documents: List[Document] = []
        filename = os.path.basename(filepath)

        try:
            doc = DocxDocument(filepath)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = "\n".join(paragraphs)

            if full_text.strip():
                documents.append(
                    Document(
                        page_content=full_text.strip(),
                        metadata=_build_document_metadata(
                            filepath,
                            "docx",
                            {
                                "file_type": "docx",
                                "paragraph_count": len(paragraphs),
                            },
                        ),
                    )
                )
        except Exception as exc:
            logger.error("Error reading DOCX '%s': %s", filename, exc)

        return documents

    def _load_txt(self, filepath: str) -> List[Document]:
        """
        Load a plain text file.

        Args:
            filepath: Path to the TXT file.

        Returns:
            List containing a single Document with full text content.
        """
        documents: List[Document] = []
        filename = os.path.basename(filepath)

        try:
            with open(filepath, "r", encoding=self.encoding) as f:
                text = f.read()

            if text.strip():
                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata=_build_document_metadata(
                            filepath,
                            "txt",
                            {"file_type": "txt"},
                        ),
                    )
                )
        except UnicodeDecodeError:
            # Fallback to latin-1 if utf-8 fails
            try:
                with open(filepath, "r", encoding="latin-1") as f:
                    text = f.read()
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text.strip(),
                            metadata=_build_document_metadata(
                                filepath,
                                "txt",
                                {"file_type": "txt", "encoding": "latin-1"},
                            ),
                        )
                    )
            except Exception as exc:
                logger.error("Error reading TXT '%s': %s", filename, exc)
        except Exception as exc:
            logger.error("Error reading TXT '%s': %s", filename, exc)

        return documents

    def _load_csv(self, filepath: str) -> List[Document]:
        """
        Load a CSV file using pandas.

        Each row is converted to a text representation and stored as
        a separate Document.

        Args:
            filepath: Path to the CSV file.

        Returns:
            List of Document objects (one per row).
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error(
                "pandas is required for CSV loading. "
                "Install: pip install pandas"
            )
            return []

        documents: List[Document] = []
        filename = os.path.basename(filepath)

        try:
            df = pd.read_csv(filepath)
            for idx, row in df.iterrows():
                # Convert row to structured text
                row_text = "\n".join(
                    f"{col}: {val}" for col, val in row.items()
                )
                documents.append(
                    Document(
                        page_content=row_text,
                        metadata=_build_document_metadata(
                            filepath,
                            "csv",
                            {
                                "row": idx,
                                "file_type": "csv",
                                "total_rows": len(df),
                            },
                        ),
                    )
                )
        except Exception as exc:
            logger.error("Error reading CSV '%s': %s", filename, exc)

        return documents

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_supported_extensions(self) -> List[str]:
        """
        Return the list of supported file extensions.

        Returns:
            List of extension strings (e.g., ['.pdf', '.docx', '.txt', '.csv']).
        """
        return list(self.supported_extensions.keys())

    def count_files(self) -> int:
        """
        Count the number of supported files in the documents directory.

        Returns:
            Integer count of supported files found.
        """
        if not os.path.isdir(self.documents_dir):
            return 0

        count = 0
        for root, _, files in os.walk(self.documents_dir):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.supported_extensions:
                    count += 1
        return count


