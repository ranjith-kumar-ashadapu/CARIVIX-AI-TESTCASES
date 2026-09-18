"""
CARIVIX AI - Data Acquisition Engine
Reusable helper functions for pulling data from APIs, websites, and files.

Covers: Government Portals / Open Data Platforms, Company Reports,
News/Blogs/Social Media, and IoT sensor feeds.
"""

import time
import json
import csv
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("carivix.data_acquisition")


def fetch_json_api(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 15,
    retries: int = 3,
    backoff_seconds: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """
    Fetch JSON data from a REST API with automatic retry and backoff.

    Use for: Government open-data APIs, financial data providers,
    census/budget report endpoints, IoT sensor feeds.

    Returns the parsed JSON dict, or None if all retries fail.
    """
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.warning("Attempt %s/%s failed for %s: %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)
    logger.error("All retries exhausted for %s", url)
    return None


def fetch_paginated_api(
    url: str,
    page_param: str = "page",
    page_size_param: str = "page_size",
    page_size: int = 100,
    max_pages: int = 50,
    headers: Optional[Dict[str, str]] = None,
    result_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch and concatenate results across multiple pages of a paginated API.

    result_key: if the API wraps results in an envelope (e.g. {"results": [...]}),
    pass the key name here. If None, assumes the response itself is a list.
    """
    all_results: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        params = {page_param: page, page_size_param: page_size}
        data = fetch_json_api(url, params=params, headers=headers)
        if not data:
            break
        page_results = data.get(result_key, []) if result_key else data
        if not page_results:
            break
        all_results.extend(page_results)
        if len(page_results) < page_size:
            break  # last page reached
    logger.info("Fetched %s total records from %s", len(all_results), url)
    return all_results


def download_file(url: str, dest_path: str, timeout: int = 30, chunk_size: int = 8192) -> bool:
    """
    Stream-download a file (PDF report, CSV export, annual report, etc.) to disk.
    Returns True on success.
    """
    try:
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        logger.info("Downloaded %s -> %s", url, dest_path)
        return True
    except requests.RequestException as exc:
        logger.error("Failed to download %s: %s", url, exc)
        return False


def poll_endpoint(
    fetch_fn: Callable[[], Optional[Dict[str, Any]]],
    interval_seconds: float,
    max_iterations: Optional[int] = None,
    on_data: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """
    Generic scheduler for real-time / near-real-time data streaming
    (e.g. traffic sensors, weather stations) when a push-based stream
    isn't available. Calls fetch_fn() on a fixed interval.

    fetch_fn: zero-arg callable returning the latest payload (or None).
    on_data: callback invoked with each successfully fetched payload.
    """
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        data = fetch_fn()
        if data and on_data:
            on_data(data)
        iteration += 1
        time.sleep(interval_seconds)


def load_local_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Load a local CSV or JSON/JSONL file into a list of dict records.
    Useful for ingesting bulk government datasets or exported reports.
    """
    file_path = Path(path)
    if file_path.suffix.lower() == ".csv":
        with open(file_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    elif file_path.suffix.lower() == ".json":
        with open(file_path, encoding="utf-8") as f:
            content = json.load(f)
            return content if isinstance(content, list) else [content]
    elif file_path.suffix.lower() == ".jsonl":
        with open(file_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def validate_record(record: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Lightweight schema check used right after collection, before a record
    is handed off to the Data Processing Engine.
    """
    missing = [field for field in required_fields if field not in record or record[field] in (None, "")]
    if missing:
        logger.warning("Record missing required fields %s: %s", missing, record)
        return False
    return True
