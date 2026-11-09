"""SQLite schema and connections.

The ledger is append-only refund rows; an order's remaining balance is derived with SUM, never stored.
`connect()` uses autocommit mode so every multi-statement write opens an explicit `BEGIN IMMEDIATE`,
which takes the write lock up front: two instances cannot both compute "remaining" against the
same snapshot and both insert.
"""
from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id     TEXT PRIMARY KEY,
    customer_id  TEXT NOT NULL,
    paid_cents   INTEGER NOT NULL CHECK (paid_cents >= 0),
    currency     TEXT NOT NULL,
    status       TEXT NOT NULL,
    paid_at      REAL,
    shipped_at   REAL,
    delivered_at REAL
);
CREATE TABLE IF NOT EXISTS refunds (
    request_id     TEXT PRIMARY KEY,                       -- client idempotency key: second line of defence
    order_id       TEXT NOT NULL REFERENCES orders(order_id),
    amount_cents   INTEGER NOT NULL CHECK (amount_cents >= 0),
    status         TEXT NOT NULL,
    fingerprint    TEXT NOT NULL,
    decision_json  TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    provider_ref   TEXT,
    attempts       INTEGER NOT NULL DEFAULT 0,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS refunds_by_order ON refunds(order_id, status);
CREATE TABLE IF NOT EXISTS policies (
    version      TEXT PRIMARY KEY,
    config_json  TEXT NOT NULL,
    activated_at REAL NOT NULL,
    active       INTEGER NOT NULL DEFAULT 0
);
"""


def connect(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False, timeout=10.0, uri=path.startswith("file:"))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    if path != ":memory:" and "mode=memory" not in path:
        conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    return conn
