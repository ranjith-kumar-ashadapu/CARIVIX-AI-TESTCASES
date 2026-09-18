"""
api_importer.py
CARIVIX AI - Data Acquisition Engine
Handles REST API data sources (e.g. open data portals, weather/IoT feeds).
Supports API-key / bearer / basic auth, retries with backoff, and
simple page-number or offset pagination.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

from importers.base_importer import BaseImporter
from utils.config_loader import ConfigLoader


class APIImporter(BaseImporter):
    """
    Fetches data from a REST API and returns it as a pandas DataFrame.

    Expected config keys (see config/data_sources.yaml -> api_sources):
        base_url, endpoint, method, auth_type, api_key_env_var,
        headers, params, timeout_seconds, retries,
        paginate, pagination {type, page_param, page_size_param, page_size}
    """

    def __init__(self, config: Dict[str, Any], source_name: str = "unnamed_source"):
        super().__init__(config, source_name)
        self.session = requests.Session()

    def _build_auth_headers(self) -> Dict[str, str]:
        cfg = self.config
        auth_type = cfg.get("auth_type", "none")
        headers = dict(cfg.get("headers", {}))

        if auth_type == "none":
            return headers

        key = ConfigLoader.resolve_env(cfg.get("api_key_env_var"))

        if auth_type == "api_key":
            headers["Authorization"] = f"Api-Key {key}"
        elif auth_type == "bearer":
            headers["Authorization"] = f"Bearer {key}"
        elif auth_type == "basic":
            # basic auth is applied via requests' `auth=` param, not headers;
            # handled separately in `_request_auth()`.
            pass
        else:
            self.logger.warning(f"Unknown auth_type '{auth_type}' — proceeding without auth.")

        return headers

    def _request_auth(self):
        cfg = self.config
        if cfg.get("auth_type") == "basic":
            key = ConfigLoader.resolve_env(cfg.get("api_key_env_var"))
            username, _, password = (key or "").partition(":")
            return HTTPBasicAuth(username, password)
        return None

    def _request_with_retries(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config
        retries = cfg.get("retries", 3)
        timeout = cfg.get("timeout_seconds", 30)
        method = cfg.get("method", "GET").upper()
        headers = self._build_auth_headers()
        auth = self._request_auth()

        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    auth=auth,
                    timeout=timeout,
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_exc = exc
                wait = 2 ** attempt  # exponential backoff
                self.logger.warning(
                    f"[{self.source_name}] Attempt {attempt}/{retries} failed: {exc}. "
                    f"Retrying in {wait}s..."
                )
                if attempt < retries:
                    time.sleep(wait)

        raise ConnectionError(
            f"[{self.source_name}] API request failed after {retries} attempts: {last_exc}"
        )

    def _read(self) -> pd.DataFrame:
        cfg = self.config
        base_url = cfg.get("base_url")
        endpoint = cfg.get("endpoint", "")
        if not base_url:
            raise ValueError("API config is missing required key 'base_url'.")

        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}" if endpoint else base_url
        base_params = dict(cfg.get("params", {}))

        if not cfg.get("paginate", False):
            payload = self._request_with_retries(url, base_params)
            records = self._unwrap_records(payload)
            return pd.json_normalize(records)

        return self._read_paginated(url, base_params)

    def _read_paginated(self, url: str, base_params: Dict[str, Any]) -> pd.DataFrame:
        cfg = self.config
        pagination = cfg.get("pagination", {}) or {}
        page_param = pagination.get("page_param", "page")
        size_param = pagination.get("page_size_param", "limit")
        page_size = pagination.get("page_size", 100)

        all_records: List[Dict[str, Any]] = []
        page = 1
        while True:
            params = dict(base_params)
            params[page_param] = page
            params[size_param] = page_size

            payload = self._request_with_retries(url, params)
            records = self._unwrap_records(payload)

            if not records:
                break

            all_records.extend(records)
            self.logger.info(f"[{self.source_name}] Fetched page {page} ({len(records)} records).")

            if len(records) < page_size:
                break  # last page
            page += 1

        return pd.json_normalize(all_records)

    @staticmethod
    def _unwrap_records(payload: Any) -> List[Dict[str, Any]]:
        """Normalize common API response shapes into a flat list of records."""
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("results", "records", "data", "items"):
                if key in payload and isinstance(payload[key], list):
                    return payload[key]
            return [payload]  # single-object response
        return []
