"""
csv_importer.py
CARIVIX AI - Data Acquisition Engine
Handles CSV data sources (e.g. census data, budget reports).
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from importers.base_importer import BaseImporter


class CSVImporter(BaseImporter):
    """
    Imports a CSV file into a pandas DataFrame based on a config dict
    such as the one produced by ConfigLoader.get_source("csv_sources", name).

    Expected config keys:
        path (str)            - required
        delimiter (str)       - default ","
        encoding (str)        - default "utf-8"
        has_header (bool)     - default True
        dtype_overrides (dict)- optional column -> dtype mapping
        parse_dates (list)    - optional list of column names to parse as dates
    """

    def _read(self) -> pd.DataFrame:
        cfg: Dict[str, Any] = self.config
        path = cfg.get("path")
        if not path:
            raise ValueError("CSV config is missing required key 'path'.")

        self._ensure_path_exists(path)

        header = 0 if cfg.get("has_header", True) else None

        df = pd.read_csv(
            path,
            delimiter=cfg.get("delimiter", ","),
            encoding=cfg.get("encoding", "utf-8"),
            header=header,
            dtype=cfg.get("dtype_overrides") or None,
            parse_dates=cfg.get("parse_dates") or None,
        )
        return df
