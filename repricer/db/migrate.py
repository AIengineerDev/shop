"""Schema application + legacy-table migration.

The brief provides ``inventory`` with REAL ``cost_price`` / ``min_margin_percent`` /
``current_price`` and a loosely-typed ``competitor_scrapes``. ``migrate()`` detects that
shape, renames the tables to ``*_legacy``, applies the canonical schema and converts rows
to integer cents / basis points. Before converting it asserts every REAL is a 2-decimal
value (up to float representation noise) so a sub-cent price such as 12.344 never silently
rounds into a wrong price.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from decimal import Decimal

from repricer.clock import iso, utcnow
from repricer.db.connection import (
    MIGRATIONS_DIR,
    apply_schema,
    split_sql,
    table_columns,
    table_exists,
    transaction,
)

LEGACY_INVENTORY_COLS = {"cost_price", "min_margin_percent", "current_price"}
MAX_LOSS = Decimal("0.000001")  # float-representation noise only; sub-cent values fail


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MigrationReport:
    migrated_legacy: bool
    inventory_rows: int
    scrape_rows: int
    applied_versions: tuple[str, ...]


def _is_legacy_inventory(conn: sqlite3.Connection) -> bool:
    if not table_exists(conn, "inventory"):
        return False
    cols = set(c.lower() for c in table_columns(conn, "inventory"))
    return bool(LEGACY_INVENTORY_COLS & cols)


def _assert_precision(conn: sqlite3.Connection) -> None:
    """Every REAL * 100 must be an integer up to float noise (MAX_LOSS)."""
    bad: list[str] = []
    rows = conn.execute(
        "SELECT sku, cost_price, min_margin_percent, current_price FROM inventory_legacy"
    ).fetchall()
    for r in rows:
        for col in ("cost_price", "min_margin_percent", "current_price"):
            val = r[col]
            if val is None:
                bad.append(f"{r['sku']}.{col} is NULL")
                continue
            d = Decimal(str(val)) * 100
            loss = abs(d - d.to_integral_value())
            if loss > MAX_LOSS:
                bad.append(f"{r['sku']}.{col}={val!r} loses {loss} after scaling")
    if bad:
        raise MigrationError("precision check failed:\n  " + "\n  ".join(bad[:20]))


def _normalize_legacy_inventory_columns(conn: sqlite3.Connection) -> None:
    """Accept SKU/sku capitalization and a missing price_class column."""
    cols = table_columns(conn, "inventory_legacy")
    lower = {c.lower(): c for c in cols}
    if "sku" in lower and lower["sku"] != "sku":
        conn.execute(f'ALTER TABLE inventory_legacy RENAME COLUMN "{lower["sku"]}" TO sku')
    if "price_class" not in lower:
        conn.execute("ALTER TABLE inventory_legacy ADD COLUMN price_class TEXT")


def _normalize_legacy_scrape_columns(conn: sqlite3.Connection) -> None:
    cols = {c.lower() for c in table_columns(conn, "competitor_scrapes_legacy")}
    for col, ddl in (
        ("scrape_id", None),
        ("sku", "TEXT"),
        ("competitor", "TEXT"),
        ("scraped_at", "TEXT"),
    ):
        if col not in cols and ddl is not None:
            conn.execute(f"ALTER TABLE competitor_scrapes_legacy ADD COLUMN {col} {ddl}")
    if "scrape_id" not in cols:
        # Fall back to rowid as the id.
        conn.execute("ALTER TABLE competitor_scrapes_legacy ADD COLUMN scrape_id INTEGER")
        conn.execute("UPDATE competitor_scrapes_legacy SET scrape_id = rowid")
    if "raw_text" not in cols:
        raise MigrationError("competitor_scrapes has no raw_text column")


def migrate(conn: sqlite3.Connection) -> MigrationReport:
    now = iso(utcnow())
    legacy = _is_legacy_inventory(conn)
    applied: list[str] = []
    with transaction(conn):
        if legacy:
            conn.execute("ALTER TABLE inventory RENAME TO inventory_legacy")
            if table_exists(conn, "competitor_scrapes"):
                conn.execute("ALTER TABLE competitor_scrapes RENAME TO competitor_scrapes_legacy")
            else:
                conn.execute(
                    "CREATE TABLE competitor_scrapes_legacy (scrape_id INTEGER, sku TEXT, "
                    "competitor TEXT, raw_text TEXT, scraped_at TEXT)"
                )
            _normalize_legacy_inventory_columns(conn)
            _normalize_legacy_scrape_columns(conn)
            _assert_precision(conn)
        apply_schema(conn)
        if legacy:
            sql = (MIGRATIONS_DIR / "0001_normalize_money.sql").read_text()
            for stmt in split_sql(sql):
                conn.execute(stmt, {"now": now})
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                ("0001_normalize_money", now),
            )
            applied.append("0001_normalize_money")
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            ("0000_schema", now),
        )
    inv = conn.execute("SELECT COUNT(*) AS n FROM inventory").fetchone()["n"]
    scr = conn.execute("SELECT COUNT(*) AS n FROM competitor_scrapes").fetchone()["n"]
    return MigrationReport(legacy, int(inv), int(scr), tuple(applied))
