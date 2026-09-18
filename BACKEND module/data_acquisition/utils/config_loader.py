"""
config_loader.py
CARIVIX AI - Data Acquisition Engine

Loads the reusable data_sources.yaml configuration and merges each
source's settings with the global defaults, so individual source
entries only need to specify overrides.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "data_sources.yaml"


class ConfigError(Exception):
    """Raised when the configuration file is missing, malformed, or a
    requested source name cannot be found."""


class ConfigLoader:
    """Loads and serves data source configuration.

    Usage:
        loader = ConfigLoader()
        csv_cfg = loader.get_source("csv_sources", "census_data")
        api_cfg = loader.get_source("api_sources", "open_data_portal")
    """

    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._raw_config: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.config_path.exists():
            raise ConfigError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            try:
                self._raw_config = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(f"Failed to parse YAML config: {exc}") from exc

    def reload(self) -> None:
        """Re-reads the config file from disk (useful if it changed at runtime)."""
        self._load()

    def get_defaults(self, category: str) -> Dict[str, Any]:
        """category: 'csv' | 'excel' | 'json' | 'api'"""
        return copy.deepcopy(self._raw_config.get("defaults", {}).get(category, {}))

    def list_sources(self, category: str) -> list[str]:
        """category: 'csv_sources' | 'excel_sources' | 'json_sources' | 'api_sources'"""
        return list(self._raw_config.get(category, {}).keys())

    def get_source(self, category: str, name: str) -> Dict[str, Any]:
        """
        Fetch a single source's config, merged with the category defaults.

        category: 'csv_sources' | 'excel_sources' | 'json_sources' | 'api_sources'
        name: the key of the source as defined in data_sources.yaml
        """
        sources = self._raw_config.get(category, {})
        if name not in sources:
            available = ", ".join(sources.keys()) or "(none defined)"
            raise ConfigError(
                f"Source '{name}' not found under '{category}'. Available: {available}"
            )

        default_key = category.replace("_sources", "")  # csv_sources -> csv
        merged = self.get_defaults(default_key)
        merged.update(copy.deepcopy(sources[name]))
        return merged

    @staticmethod
    def resolve_env(var_name: str | None) -> str | None:
        """Resolve an API key / token from an environment variable name."""
        if not var_name:
            return None
        value = os.environ.get(var_name)
        if value is None:
            raise ConfigError(
                f"Environment variable '{var_name}' is not set. "
                f"Export it before running the importer, e.g.: export {var_name}=your_key"
            )
        return value
