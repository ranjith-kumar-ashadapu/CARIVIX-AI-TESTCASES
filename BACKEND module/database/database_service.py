from __future__ import annotations
from typing import Any, Optional, Sequence

from db_crud import DatabaseManager
from logging_config import get_logger

logger = get_logger("carivix.database_service")


class DatabaseServiceError(Exception):
    """Raised when a database read/write operation fails."""


class DatabaseService:
    """
    Connects the CARIVIX AI data pipeline to persistent storage.
    Wraps DatabaseManager with logging and consistent error handling,
    so calling code doesn't need to manage sqlite3 exceptions directly.
    """

    def __init__(self, db_path: str = "carivix.db"):
        self.db_path = db_path
        try:
            self.db = DatabaseManager(db_path)
            logger.info(f"Connected to database at '{db_path}'", extra={"context": "db_connect"})
        except Exception as exc:
            logger.error(f"Failed to connect to database at '{db_path}': {exc}", extra={"context": "db_connect"})
            raise DatabaseServiceError(f"Could not connect to database: {exc}") from exc

    def write(self, table: str, records: Sequence[dict]) -> int:
        """Insert one or more records into a table."""
        records = [
            {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in r.items()}
            for r in records
        ]
        try:
            if len(records) == 1:
                new_id = self.db.insert(table, records[0])
                logger.info(f"Wrote 1 record to '{table}' (id={new_id})", extra={"context": "db_write"})
                return 1
            count = self.db.insert_many(table, records)
            logger.info(f"Wrote {count} record(s) to '{table}'", extra={"context": "db_write"})
            return count
        except Exception as exc:
            logger.error(f"Write failed on table '{table}': {exc}", extra={"context": "db_write"})
            raise DatabaseServiceError(f"Write failed on '{table}': {exc}") from exc

    def read(self, table: str, where: Optional[dict] = None) -> list[dict]:
        """Read records from a table, optionally filtered."""
        try:
            rows = self.db.read(table, where=where)
            logger.info(f"Read {len(rows)} record(s) from '{table}'", extra={"context": "db_read"})
            return rows
        except Exception as exc:
            logger.error(f"Read failed on table '{table}': {exc}", extra={"context": "db_read"})
            raise DatabaseServiceError(f"Read failed on '{table}': {exc}") from exc

    def close(self) -> None:
        self.db.close()
        logger.info(f"Closed database connection at '{self.db_path}'", extra={"context": "db_connect"})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()