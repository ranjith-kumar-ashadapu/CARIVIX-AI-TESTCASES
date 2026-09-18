"""
db_crud.py
Reusable database CRUD (Create, Read, Update, Delete) integration layer.

Uses sqlite3 by default (zero-config, file-based) but is written so the
underlying connection can be swapped for psycopg2 (PostgreSQL), mysql-connector,
or any DB-API 2.0-compliant driver with minimal changes -- the SQL here uses
'?' placeholders (sqlite style); for PostgreSQL/MySQL swap to '%s' placeholders.

Usage:
    db = DatabaseManager("data/app.db")
    db.create_table("users", {"id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                               "name": "TEXT NOT NULL",
                               "email": "TEXT UNIQUE"})
    db.insert("users", {"name": "Sindhu", "email": "sindhu@example.com"})
    rows = db.read("users", where={"name": "Sindhu"})
    db.update("users", {"email": "new@example.com"}, where={"id": 1})
    db.delete("users", where={"id": 1})
"""

from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from typing import Any, Optional, Sequence


class DatabaseManager:
    """Lightweight, reusable CRUD wrapper around a sqlite3 connection."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")

    # -- connection lifecycle -------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @contextmanager
    def transaction(self):
        """Group multiple statements into one atomic commit; rolls back on error."""
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # -- schema helpers ---------------------------------------------------------
    def create_table(self, table: str, columns: dict[str, str], if_not_exists: bool = True) -> None:
        """columns: {"col_name": "SQL TYPE + constraints"}"""
        cols_sql = ", ".join(f"{name} {dtype}" for name, dtype in columns.items())
        exists_clause = "IF NOT EXISTS " if if_not_exists else ""
        sql = f"CREATE TABLE {exists_clause}{table} ({cols_sql})"
        self._conn.execute(sql)
        self._conn.commit()

    def drop_table(self, table: str, if_exists: bool = True) -> None:
        exists_clause = "IF EXISTS " if if_exists else ""
        self._conn.execute(f"DROP TABLE {exists_clause}{table}")
        self._conn.commit()

    # -- CREATE -------------------------------------------------------------
    def insert(self, table: str, record: dict[str, Any]) -> int:
        """Insert a single row. Returns the new row's id (lastrowid)."""
        cols = ", ".join(record.keys())
        placeholders = ", ".join("?" for _ in record)
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        cur = self._conn.execute(sql, tuple(record.values()))
        self._conn.commit()
        return cur.lastrowid

    def insert_many(self, table: str, records: Sequence[dict[str, Any]]) -> int:
        """Bulk insert. All dicts must share the same keys. Returns rows affected."""
        if not records:
            return 0
        cols = ", ".join(records[0].keys())
        placeholders = ", ".join("?" for _ in records[0])
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        rows = [tuple(r.values()) for r in records]
        cur = self._conn.executemany(sql, rows)
        self._conn.commit()
        return cur.rowcount

    # -- READ -----------------------------------------------------------------
    def read(self, table: str, columns: Sequence[str] = ("*",),
              where: Optional[dict[str, Any]] = None, order_by: str = None,
              limit: int = None) -> list[dict]:
        """Fetch rows as a list of dicts. `where` is an equality-AND filter."""
        cols = ", ".join(columns)
        sql = f"SELECT {cols} FROM {table}"
        params: tuple = ()
        if where:
            clause = " AND ".join(f"{k} = ?" for k in where)
            sql += f" WHERE {clause}"
            params = tuple(where.values())
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit:
            sql += f" LIMIT {limit}"
        cur = self._conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def read_one(self, table: str, where: dict[str, Any], columns: Sequence[str] = ("*",)) -> Optional[dict]:
        rows = self.read(table, columns=columns, where=where, limit=1)
        return rows[0] if rows else None

    def count(self, table: str, where: Optional[dict[str, Any]] = None) -> int:
        sql = f"SELECT COUNT(*) as n FROM {table}"
        params: tuple = ()
        if where:
            clause = " AND ".join(f"{k} = ?" for k in where)
            sql += f" WHERE {clause}"
            params = tuple(where.values())
        return self._conn.execute(sql, params).fetchone()["n"]

    # -- UPDATE ---------------------------------------------------------------
    def update(self, table: str, changes: dict[str, Any], where: dict[str, Any]) -> int:
        """Update rows matching `where` with `changes`. Returns rows affected."""
        set_clause = ", ".join(f"{k} = ?" for k in changes)
        where_clause = " AND ".join(f"{k} = ?" for k in where)
        sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        params = tuple(changes.values()) + tuple(where.values())
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return cur.rowcount

    # -- DELETE ---------------------------------------------------------------
    def delete(self, table: str, where: dict[str, Any]) -> int:
        """Delete rows matching `where`. Returns rows affected. Refuses if `where` is empty (safety guard)."""
        if not where:
            raise ValueError("delete() requires a non-empty `where` clause to avoid wiping the table.")
        where_clause = " AND ".join(f"{k} = ?" for k in where)
        sql = f"DELETE FROM {table} WHERE {where_clause}"
        cur = self._conn.execute(sql, tuple(where.values()))
        self._conn.commit()
        return cur.rowcount

    # -- raw escape hatch -------------------------------------------------------
    def execute(self, sql: str, params: tuple = ()) -> list[dict]:
        """For anything not covered above (joins, aggregates, etc). Always use '?' placeholders."""
        cur = self._conn.execute(sql, params)
        if cur.description:
            return [dict(row) for row in cur.fetchall()]
        self._conn.commit()
        return []
