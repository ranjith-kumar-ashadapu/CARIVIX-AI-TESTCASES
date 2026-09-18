"""
CARIVIX AI - Data Processing Engine
Logging Configuration Module
Author: Sindhu Bollikonda (Python/R Developer)

Provides a centralized, reusable logging setup used across the Data
Acquisition, Data Processing, and Validation layers of the CARIVIX platform.

Features:
    - Console logging (human-readable, colored levels for local dev)
    - Rotating file logging (size-based rotation, avoids unbounded log growth)
    - JSON-formatted log records optionally, for downstream log aggregation
      (e.g. ELK / CloudWatch) as referenced in the CARIVIX Technology Stack
    - Per-module logger factory so every module logs under its own name
    - Configurable via environment variables so no code changes are needed
      between dev / staging / production
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone

# --------------------------------------------------------------------------- #
# Configuration (overridable via environment variables)
# --------------------------------------------------------------------------- #
LOG_DIR = os.environ.get("CARIVIX_LOG_DIR", os.path.join(os.path.dirname(__file__), "logs"))
LOG_FILE = os.environ.get("CARIVIX_LOG_FILE", "carivix_pipeline.log")
LOG_LEVEL = os.environ.get("CARIVIX_LOG_LEVEL", "INFO").upper()
LOG_MAX_BYTES = int(os.environ.get("CARIVIX_LOG_MAX_BYTES", 5 * 1024 * 1024))  # 5 MB
LOG_BACKUP_COUNT = int(os.environ.get("CARIVIX_LOG_BACKUP_COUNT", 5))
LOG_AS_JSON = os.environ.get("CARIVIX_LOG_JSON", "false").lower() == "true"

os.makedirs(LOG_DIR, exist_ok=True)


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON, useful for log aggregation
    tools referenced in the CARIVIX architecture (e.g. cloud log pipelines)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class ContextFilter(logging.Filter):
    """Injects a default 'context' attribute so format strings referencing
    %(context)s never crash, even when the caller doesn't supply one."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "context"):
            record.context = "general"
        return True


_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | [%(context)s] | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured_loggers = set()


def get_logger(name: str) -> logging.Logger:
    """
    Returns a fully-configured logger for the given module name.

    Usage:
        from logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Data validation started", extra={"context": "validation"})

    Idempotent: safe to call multiple times for the same name without
    duplicating handlers (a common source of duplicate log lines).
    """
    logger = logging.getLogger(name)

    if name in _configured_loggers:
        return logger

    logger.setLevel(LOG_LEVEL)
    logger.propagate = False
    logger.addFilter(ContextFilter())

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(console_handler)

    # --- Rotating file handler ---
    file_path = os.path.join(LOG_DIR, LOG_FILE)
    file_handler = logging.handlers.RotatingFileHandler(
        file_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(LOG_LEVEL)
    if LOG_AS_JSON:
        file_handler.setFormatter(JsonFormatter())
    else:
        file_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(file_handler)

    _configured_loggers.add(name)
    return logger


def log_exception(logger: logging.Logger, message: str, exc: Exception, context: str = "error") -> None:
    """Standardized way to log caught exceptions with full traceback."""
    logger.error(f"{message}: {exc}", exc_info=True, extra={"context": context})


if __name__ == "__main__":
    # Quick self-test
    test_logger = get_logger("carivix.logging_config.selftest")
    test_logger.debug("Debug message (only visible if LOG_LEVEL=DEBUG)")
    test_logger.info("Logging module initialized successfully", extra={"context": "startup"})
    test_logger.warning("This is a sample warning", extra={"context": "startup"})
    try:
        1 / 0
    except ZeroDivisionError as e:
        log_exception(test_logger, "Self-test caught an expected error", e)
    print(f"\nLog file written to: {os.path.join(LOG_DIR, LOG_FILE)}")
