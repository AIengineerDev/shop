from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from repricer.db.connection import apply_schema, connect
from repricer.db.migrate import migrate
from repricer.seed import generate, write_legacy_db

FROZEN_NOW = datetime(2026, 11, 11, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def frozen_now() -> datetime:
    return FROZEN_NOW


@pytest.fixture
def tmp_db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """Fresh SQLite file with the canonical schema applied and no data."""
    conn = connect(tmp_path / "test.db")
    apply_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def seeded_db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """Legacy-schema seed written, then migrated. Deterministic (seed 42)."""
    conn = connect(tmp_path / "seeded.db")
    write_legacy_db(conn, generate(seed=42, now=FROZEN_NOW))
    migrate(conn)
    try:
        yield conn
    finally:
        conn.close()
