from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from repricer.db.connection import connect, table_columns
from repricer.db.migrate import MigrationError, migrate
from repricer.seed import generate, write_legacy_db

NOW = "2026-11-11T12:00:00.000Z"


def _seed_run(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO repricing_runs VALUES ('r1', ?, NULL, 'test', 1, 0, 0, 0, 'running', NULL)",
        (NOW,),
    )
    conn.execute(
        "INSERT INTO inventory VALUES ('S1','t',3000,2000,4500,5,'default',?,?)", (NOW, NOW)
    )


def test_price_changes_append_only(tmp_db: sqlite3.Connection) -> None:
    _seed_run(tmp_db)
    tmp_db.execute(
        "INSERT INTO price_changes VALUES ('c1','S1',NULL,NULL,4500,4099,'seed','t',?,NULL,'h')",
        (NOW,),
    )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        tmp_db.execute("UPDATE price_changes SET new_price_cents = 1 WHERE change_id = 'c1'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        tmp_db.execute("DELETE FROM price_changes WHERE change_id = 'c1'")


def test_review_actions_append_only(tmp_db: sqlite3.Connection) -> None:
    _seed_run(tmp_db)
    tmp_db.execute(
        "INSERT INTO price_recommendations VALUES ('rec1','r1','S1',4500,3000,2000,3600,NULL,'[]',"
        "4500,4500,0,0,'hold','[]','x',1,'pending',?)",
        (NOW,),
    )
    tmp_db.execute(
        "INSERT INTO review_actions VALUES ('a1','rec1','approve','alex',NULL,NULL,?)", (NOW,)
    )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        tmp_db.execute("UPDATE review_actions SET actor = 'mallory' WHERE action_id = 'a1'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        tmp_db.execute("DELETE FROM review_actions")


def test_recommendation_only_review_status_mutable(tmp_db: sqlite3.Connection) -> None:
    _seed_run(tmp_db)
    tmp_db.execute(
        "INSERT INTO price_recommendations VALUES ('rec1','r1','S1',4500,3000,2000,3600,NULL,'[]',"
        "4500,4099,-401,-891,'propose','[]','x',1,'pending',?)",
        (NOW,),
    )
    tmp_db.execute(
        "UPDATE price_recommendations SET review_status='approved' WHERE recommendation_id='rec1'"
    )
    with pytest.raises(sqlite3.IntegrityError, match="only review_status"):
        tmp_db.execute(
            "UPDATE price_recommendations SET proposed_price_cents=1 WHERE recommendation_id='rec1'"
        )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        tmp_db.execute("DELETE FROM price_recommendations")


def test_migration_converts_legacy_to_cents(tmp_path) -> None:  # type: ignore[no-untyped-def]
    conn = connect(tmp_path / "legacy.db")
    data = generate(seed=42, n_skus=5, n_scrapes=10, now=datetime(2026, 11, 11, tzinfo=UTC))
    write_legacy_db(conn, data)
    assert "cost_price" in table_columns(conn, "inventory")
    report = migrate(conn)
    assert report.migrated_legacy
    assert report.inventory_rows == 5
    assert report.scrape_rows == 10
    cols = table_columns(conn, "inventory")
    assert "cost_price_cents" in cols and "cost_price" not in cols
    row = conn.execute("SELECT * FROM inventory WHERE sku='SKU-0001'").fetchone()
    assert row["cost_price_cents"] == 3000
    assert row["min_margin_bp"] == 2000
    assert row["current_price_cents"] == 4500
    # idempotent
    report2 = migrate(conn)
    assert not report2.migrated_legacy
    assert report2.inventory_rows == 5


def test_migration_rejects_precision_loss(tmp_path) -> None:  # type: ignore[no-untyped-def]
    conn = connect(tmp_path / "bad.db")
    data = generate(seed=1, n_skus=2, n_scrapes=2)
    write_legacy_db(conn, data)
    # sub-cent value: must be refused, never rounded
    conn.execute("UPDATE inventory SET cost_price = 12.344 WHERE SKU = 'SKU-0001'")
    with pytest.raises(MigrationError, match="precision"):
        migrate(conn)
    # nothing half-applied
    assert "cost_price" in table_columns(conn, "inventory")


def test_seed_is_deterministic() -> None:
    a = generate(seed=42, now=datetime(2026, 1, 1, tzinfo=UTC))
    b = generate(seed=42, now=datetime(2026, 1, 1, tzinfo=UTC))
    assert [s.raw_text for s in a.scrapes] == [s.raw_text for s in b.scrapes]
    assert a.inventory[0].sku == "SKU-0001"
    assert a.inventory[0].cost_price == Decimal("30.00")
    assert len(a.scrapes) == 200
    assert len(a.inventory) == 50
