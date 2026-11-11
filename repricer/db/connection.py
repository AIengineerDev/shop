"""SQLite connection + schema management."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), isolation_level=None)  # autocommit; explicit BEGIN
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def split_sql(script: str) -> list[str]:
    """Split a script into complete statements (trigger bodies included)."""
    out: list[str] = []
    buf: list[str] = []
    for line in script.splitlines():
        if not buf and (not line.strip() or line.strip().startswith("--")):
            continue
        buf.append(line)
        if sqlite3.complete_statement("\n".join(buf)):
            out.append("\n".join(buf).strip())
            buf = []
    tail = "\n".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def apply_schema(conn: sqlite3.Connection) -> None:
    """Statement-by-statement so it can run inside a caller's transaction.

    ``executescript`` would COMMIT first, which breaks migration atomicity.
    """
    for stmt in split_sql(SCHEMA_PATH.read_text()):
        conn.execute(stmt)


@contextmanager
def transaction(conn: sqlite3.Connection, immediate: bool = True) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row["name"] for row in conn.execute(f"PRAGMA table_info({table})")]


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None
