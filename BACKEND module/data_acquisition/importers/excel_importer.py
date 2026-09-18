"""
excel_importer.py
CARIVIX AI - Data Acquisition Engine
Handles Excel data sources (e.g. company financials, annual reports).
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from importers.base_importer import BaseImporter


class ExcelImporter(BaseImporter):
    """
    Imports an Excel workbook into a pandas DataFrame (or a dict of
    DataFrames if sheet_name == "all").

    Expected config keys:
        path (str)             - required
        sheet_name (str|int)   - default 0; use "all" to load every sheet
        header_row (int)       - default 0
        skip_rows (int)        - default 0
        dtype_overrides (dict) - optional column -> dtype mapping
    """

    def _read(self):  # -> pd.DataFrame | Dict[str, pd.DataFrame]
        cfg: Dict[str, Any] = self.config
        path = cfg.get("path")
        if not path:
            raise ValueError("Excel config is missing required key 'path'.")

        self._ensure_path_exists(path)

        sheet_name = cfg.get("sheet_name", 0)
        read_all_sheets = isinstance(sheet_name, str) and sheet_name.lower() == "all"

        result = pd.read_excel(
            path,
            sheet_name=None if read_all_sheets else sheet_name,
            header=cfg.get("header_row", 0),
            skiprows=cfg.get("skip_rows", 0) or None,
            dtype=cfg.get("dtype_overrides") or None,
        )
        return result

    def _validate(self, df) -> None:
        # When all sheets are loaded, `df` is a dict of {sheet_name: DataFrame}.
        if isinstance(df, dict):
            if not df:
                raise ValueError(f"[{self.source_name}] Workbook contained no sheets.")
            for sheet, frame in df.items():
                if frame.empty:
                    self.logger.warning(f"[{self.source_name}] Sheet '{sheet}' is empty.")
        else:
            super()._validate(df)

    def load(self, validate: bool = True):
        """Overrides base load() to log sheet-level detail when multiple sheets are read."""
        self.logger.info(f"Loading source '{self.source_name}' ...")
        try:
            result = self._read()
        except FileNotFoundError as exc:
            from importers.base_importer import ImportError_
            raise ImportError_(f"[{self.source_name}] File not found: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            from importers.base_importer import ImportError_
            raise ImportError_(f"[{self.source_name}] Failed to read source: {exc}") from exc

        if validate:
            self._validate(result)

        if isinstance(result, dict):
            summary = ", ".join(f"{name}({len(df)} rows)" for name, df in result.items())
            self.logger.info(f"Loaded '{self.source_name}' — sheets: {summary}")
        else:
            self.logger.info(
                f"Loaded '{self.source_name}': {len(result)} rows, {len(result.columns)} columns."
            )
        return result
