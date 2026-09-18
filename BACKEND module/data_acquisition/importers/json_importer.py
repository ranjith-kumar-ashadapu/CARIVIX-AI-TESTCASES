"""
json_importer.py
CARIVIX AI - Data Acquisition Engine
Handles JSON data sources (e.g. government open data, scraped news articles).
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd

from importers.base_importer import BaseImporter


class JSONImporter(BaseImporter):
    """
    Imports a JSON file into a pandas DataFrame, with support for
    nested records and optional flattening (normalization).

    Expected config keys:
        path (str)          - required
        encoding (str)      - default "utf-8"
        record_path (str)   - optional dotted path to the list of records,
                               e.g. "results.records" for {"results": {"records": [...]}}
        normalize (bool)    - default True; flattens nested dicts into columns
    """

    def _read(self) -> pd.DataFrame:
        cfg: Dict[str, Any] = self.config
        path = cfg.get("path")
        if not path:
            raise ValueError("JSON config is missing required key 'path'.")

        self._ensure_path_exists(path)

        with open(path, "r", encoding=cfg.get("encoding", "utf-8")) as f:
            raw = json.load(f)

        record_path = cfg.get("record_path")
        records = self._extract_records(raw, record_path)

        if cfg.get("normalize", True):
            df = pd.json_normalize(records)
        else:
            df = pd.DataFrame(records)

        return df

    @staticmethod
    def _extract_records(raw: Any, record_path: str | None) -> Any:
        """Walk a dotted path (e.g. 'results.records') into nested JSON."""
        if not record_path:
            return raw

        node = raw
        for key in record_path.split("."):
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                raise KeyError(
                    f"record_path '{record_path}' is invalid — key '{key}' not found."
                )
        return node
