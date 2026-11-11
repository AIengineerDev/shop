from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from repricer.db import repo
from repricer.execution import apply as apply_mod
from repricer.execution import review
from repricer.execution.errors import StaleSnapshot
from tests.unit.helpers import add_inventory, recommend


def test_ac_x1_atomic_apply(
    tmp_db: sqlite3.Connection, frozen_now: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_inventory(tmp_db, "S1", frozen_now)
    rec_id = recommend(tmp_db, "S1", frozen_now)

    def boom(*a: object, **k: object) -> None:
        raise RuntimeError("injected between insert_change and price update")

    monkeypatch.setattr(apply_mod.repo, "_set_inventory_price", boom)
    with pytest.raises(RuntimeError, match="injected"):
        review.approve(tmp_db, rec_id, actor="a", note=None, now=frozen_now, hard_cap_bp=5000)
    # everything rolled back: no change row, price untouched, rec back to pending, no action
    assert repo.changes_for_sku(tmp_db, "S1") == []
    assert repo.get_inventory(tmp_db, "S1").current_price_cents == 4500
    assert repo.get_recommendation(tmp_db, rec_id)["review_status"] == "pending"
    assert repo.actions_for_recommendation(tmp_db, rec_id) == []


def test_ac_x4_stale_snapshot_expires(tmp_db: sqlite3.Connection, frozen_now: datetime) -> None:
    add_inventory(tmp_db, "S1", frozen_now)
    rec_id = recommend(tmp_db, "S1", frozen_now)
    # price moved underneath (e.g. another channel)
    tmp_db.execute("UPDATE inventory SET current_price_cents = 4700 WHERE sku = 'S1'")
    with pytest.raises(StaleSnapshot):
        review.approve(tmp_db, rec_id, actor="a", note=None, now=frozen_now, hard_cap_bp=5000)
    assert repo.get_recommendation(tmp_db, rec_id)["review_status"] == "expired"
    assert repo.changes_for_sku(tmp_db, "S1") == []
    assert repo.get_inventory(tmp_db, "S1").current_price_cents == 4700


def test_apply_revalidates_against_live_cost(
    tmp_db: sqlite3.Connection, frozen_now: datetime
) -> None:
    add_inventory(tmp_db, "S1", frozen_now)
    rec_id = recommend(tmp_db, "S1", frozen_now)  # proposes 40.99
    # cost jumped so floor is now 48.00 > proposed; apply must refuse
    tmp_db.execute("UPDATE inventory SET cost_price_cents = 4000 WHERE sku = 'S1'")
    from repricer.execution.errors import FloorViolation

    with pytest.raises(FloorViolation):
        review.approve(tmp_db, rec_id, actor="a", note=None, now=frozen_now, hard_cap_bp=5000)
    assert repo.get_inventory(tmp_db, "S1").current_price_cents == 4500
    assert repo.get_recommendation(tmp_db, rec_id)["review_status"] == "pending"


def test_new_run_expires_pending(tmp_db: sqlite3.Connection, frozen_now: datetime) -> None:
    add_inventory(tmp_db, "S1", frozen_now)
    first = recommend(tmp_db, "S1", frozen_now)
    from repricer.db.connection import transaction

    with transaction(tmp_db):
        assert repo.expire_pending_for_sku(tmp_db, "S1") == 1
    assert repo.get_recommendation(tmp_db, first)["review_status"] == "expired"
