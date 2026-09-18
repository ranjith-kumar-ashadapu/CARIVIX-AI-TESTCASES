from __future__ import annotations
from typing import Any, Optional

from importers import get_importer
from utils.config_loader import ConfigLoader

VALID_CATEGORIES = {"csv_sources", "excel_sources", "json_sources", "api_sources"}


class IngestionValidationError(ValueError):
    """Raised when ingest() is called with an invalid category or source name."""


class DataIngestionService:
    """
    Single entry point for pulling data into CARIVIX AI, regardless of
    source type. Wraps config loading and importer selection so calling
    code only needs a category and a source name.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.loader = ConfigLoader(config_path)

    def _validate_inputs(self, category: str, name: str) -> None:
        if not category or not isinstance(category, str):
            raise IngestionValidationError("category must be a non-empty string")
        if category not in VALID_CATEGORIES:
            raise IngestionValidationError(
                f"Unknown category '{category}'. Expected one of: {sorted(VALID_CATEGORIES)}"
            )
        if not name or not isinstance(name, str):
            raise IngestionValidationError("name must be a non-empty string")

        available = self.loader.list_sources(category)
        if name not in available:
            raise IngestionValidationError(
                f"Source '{name}' not found under '{category}'. Available: {available}"
            )

    def ingest(self, category: str, name: str) -> Any:
        self._validate_inputs(category, name)
        source_cfg = self.loader.get_source(category, name)
        importer = get_importer(category, source_cfg, source_name=name)
        return importer.load()

    def list_sources(self, category: Optional[str] = None) -> dict:
        categories = [category] if category else sorted(VALID_CATEGORIES)
        return {cat: self.loader.list_sources(cat) for cat in categories}