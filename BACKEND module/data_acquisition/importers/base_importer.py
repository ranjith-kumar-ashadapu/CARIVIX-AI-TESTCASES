"""
base_importer.py
CARIVIX AI - Data Acquisition Engine

Abstract base class that all data importers (CSV, Excel, JSON, API)
inherit from. Guarantees a consistent .load() -> pandas.DataFrame
interface and shared validation / error-handling behavior across the
whole Data Acquisition Engine (see Module 1 in the CARIVIX AI spec).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from utils.logger import get_logger


class ImportError_(Exception):
    """Raised when a source cannot be loaded or fails validation."""


class BaseImporter(ABC):
    """
    Common contract for every importer.

    Subclasses must implement `_read()`, which returns a pandas
    DataFrame. `load()` wraps `_read()` with logging, timing, and
    basic post-load validation so behavior is identical across
    source types.
    """

    def __init__(self, config: Dict[str, Any], source_name: str = "unnamed_source"):
        self.config = config
        self.source_name = source_name
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def _read(self) -> pd.DataFrame:
        """Subclasses implement the actual read logic here."""
        raise NotImplementedError

    def load(self, validate: bool = True) -> pd.DataFrame:
        """Load the source into a DataFrame, with logging and validation."""
        self.logger.info(f"Loading source '{self.source_name}' ...")
        try:
            df = self._read()
        except FileNotFoundError as exc:
            raise ImportError_(f"[{self.source_name}] File not found: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - surface all read failures uniformly
            raise ImportError_(f"[{self.source_name}] Failed to read source: {exc}") from exc

        if validate:
            self._validate(df)

        self.logger.info(
            f"Loaded '{self.source_name}': {len(df)} rows, {len(df.columns)} columns."
        )
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        """Basic sanity checks applied to every source. Subclasses may extend."""
        if df is None:
            raise ImportError_(f"[{self.source_name}] Importer returned None instead of a DataFrame.")
        if df.empty:
            self.logger.warning(f"[{self.source_name}] Loaded DataFrame is empty.")

    @staticmethod
    def _ensure_path_exists(path: str) -> Path:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(str(resolved))
        return resolved
