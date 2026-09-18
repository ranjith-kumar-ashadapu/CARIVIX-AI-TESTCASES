"""
importers package
CARIVIX AI - Data Acquisition Engine

Exposes a factory function `get_importer()` so calling code doesn't
need to know which concrete importer class to instantiate.
"""

from typing import Any, Dict

from importers.api_importer import APIImporter
from importers.base_importer import BaseImporter
from importers.csv_importer import CSVImporter
from importers.excel_importer import ExcelImporter
from importers.json_importer import JSONImporter

_CATEGORY_TO_IMPORTER = {
    "csv_sources": CSVImporter,
    "excel_sources": ExcelImporter,
    "json_sources": JSONImporter,
    "api_sources": APIImporter,
}


def get_importer(category: str, config: Dict[str, Any], source_name: str) -> BaseImporter:
    """
    Factory: returns the correct importer instance for a given category.

    category: 'csv_sources' | 'excel_sources' | 'json_sources' | 'api_sources'
    """
    importer_cls = _CATEGORY_TO_IMPORTER.get(category)
    if importer_cls is None:
        raise ValueError(
            f"Unknown category '{category}'. Expected one of: "
            f"{list(_CATEGORY_TO_IMPORTER.keys())}"
        )
    return importer_cls(config=config, source_name=source_name)


__all__ = [
    "BaseImporter",
    "CSVImporter",
    "ExcelImporter",
    "JSONImporter",
    "APIImporter",
    "get_importer",
]
