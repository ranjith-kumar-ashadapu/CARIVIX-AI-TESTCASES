"""
CARIVIX AI - Data Processing Engine
Error Handling Module
Author: Sindhu Bollikonda (Python/R Developer)

Provides a consistent error-handling framework across the CARIVIX Data
Acquisition, Data Processing, and AI Intelligence layers:

    - A hierarchy of custom exceptions so calling code can catch specific
      failure types (data errors vs. source/connection errors vs. config
      errors) instead of bare `except Exception`.
    - A `@safe_run` decorator that logs and optionally retries failed
      operations (useful for flaky data sources: APIs, web scraping, IoT
      sensor feeds mentioned in the CARIVIX Data Acquisition Engine).
    - An `ErrorCollector` for batch/pipeline jobs that should keep going
      and report *all* row-level errors at the end, rather than crashing
      on the first bad record.
"""

from __future__ import annotations

import functools
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Type

from logging_config import get_logger, log_exception

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Custom Exception Hierarchy
# --------------------------------------------------------------------------- #
class CarivixError(Exception):
    """Base class for all CARIVIX platform errors."""


class DataValidationError(CarivixError):
    """Raised when a dataset or record fails validation rules."""


class DataSourceError(CarivixError):
    """Raised when a data source (API, scraper, sensor feed, DB) fails
    to return usable data — e.g. connection failures, timeouts, HTTP errors."""


class SchemaError(CarivixError):
    """Raised when incoming data does not match the expected schema
    (missing columns, wrong types, unexpected structure)."""


class ConfigurationError(CarivixError):
    """Raised for missing or invalid configuration (env vars, credentials,
    malformed pipeline settings)."""


class ProcessingError(CarivixError):
    """Raised for failures during transformation / feature engineering
    steps in the Data Processing Engine."""


# --------------------------------------------------------------------------- #
# Retry / Safe-Execution Decorator
# --------------------------------------------------------------------------- #
def safe_run(
    retries: int = 0,
    delay_seconds: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[Type[BaseException], ...] = (Exception,),
    reraise: bool = True,
    default: Any = None,
):
    """
    Decorator that wraps a function with logging + optional retry/backoff.

    Args:
        retries: number of retry attempts after the initial try (0 = no retry)
        delay_seconds: initial delay between retries
        backoff: multiplier applied to delay after each failed attempt
        exceptions: exception types that should trigger a retry
        reraise: if True, re-raises the final exception after retries are
                 exhausted; if False, returns `default` instead
        default: value returned when reraise=False and all attempts fail

    Example:
        @safe_run(retries=3, delay_seconds=2, exceptions=(DataSourceError,))
        def fetch_from_api(url):
            ...
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay_seconds
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    attempt += 1
                    if attempt > retries:
                        log_exception(
                            logger,
                            f"'{func.__name__}' failed after {attempt} attempt(s)",
                            exc,
                            context="safe_run",
                        )
                        if reraise:
                            raise
                        return default
                    logger.warning(
                        f"'{func.__name__}' failed (attempt {attempt}/{retries}): {exc}. "
                        f"Retrying in {current_delay:.1f}s...",
                        extra={"context": "safe_run.retry"},
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# Batch Error Collector
# --------------------------------------------------------------------------- #
@dataclass
class ErrorRecord:
    index: Any
    message: str
    exception_type: str
    traceback: Optional[str] = None


@dataclass
class ErrorCollector:
    """
    Accumulates errors encountered while processing a batch/dataset so a
    pipeline job can continue past bad rows and report a full summary at
    the end, instead of failing on the first exception.

    Example:
        collector = ErrorCollector()
        for i, row in enumerate(dataset):
            try:
                process(row)
            except CarivixError as e:
                collector.add(i, e)

        if collector.has_errors():
            collector.log_summary(logger)
    """

    errors: List[ErrorRecord] = field(default_factory=list)

    def add(self, index: Any, exc: Exception, include_traceback: bool = False) -> None:
        self.errors.append(
            ErrorRecord(
                index=index,
                message=str(exc),
                exception_type=type(exc).__name__,
                traceback=traceback.format_exc() if include_traceback else None,
            )
        )

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def count(self) -> int:
        return len(self.errors)

    def as_dicts(self) -> List[dict]:
        return [e.__dict__ for e in self.errors]

    def log_summary(self, log) -> None:
        log.error(
            f"Batch completed with {self.count()} error(s) out of processing run.",
            extra={"context": "error_collector"},
        )
        for err in self.errors[:20]:  # cap verbose logging
            log.error(
                f"  Row {err.index}: [{err.exception_type}] {err.message}",
                extra={"context": "error_collector"},
            )
        if self.count() > 20:
            log.error(f"  ... and {self.count() - 20} more error(s) not shown.")


if __name__ == "__main__":
    # Self-test
    @safe_run(retries=2, delay_seconds=0.1, exceptions=(DataSourceError,), reraise=False, default="FALLBACK")
    def flaky_source(fail_times: int, _state={"calls": 0}):
        _state["calls"] += 1
        if _state["calls"] <= fail_times:
            raise DataSourceError("Simulated transient failure")
        return "OK"

    result = flaky_source(2)
    print("safe_run result:", result)

    collector = ErrorCollector()
    sample_rows = [1, -5, "bad", 42]
    for idx, val in enumerate(sample_rows):
        try:
            if not isinstance(val, int) or val < 0:
                raise DataValidationError(f"Invalid value: {val!r}")
        except DataValidationError as e:
            collector.add(idx, e)

    if collector.has_errors():
        collector.log_summary(logger)
    print("Total errors:", collector.count())
