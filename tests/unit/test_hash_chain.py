from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta

from repricer.db import repo
from repricer.execution import review
from repricer.execution.audit import history, verify_chain
from tests.unit.helpers import add_inventory, recommend


def test_ac_x2_chain_verifies_then_detects_tamper(
    tmp_db: sqlite3.Connection, frozen_now: datetime
) -> None:
    rng = random.Random(7)
    skus = [f"S{i}" for i in range(5)]
    for s in skus:
        add_inventory(tmp_db, s, frozen_now, cost=2000, margin_bp=1000, current=5000)
    approved = 0
    t = frozen_now
    while approved < 100:
        sku = rng.choice(skus)
        inv = repo.get_inventory(tmp_db, sku)
        # alternate above/below so bounds and floor allow movement
        factor = rng.choice([0.93, 1.04])
        target = int(inv.current_price_cents * factor)
        rec_id = recommend(tmp_db, sku, t, competitor_prices=(target, target + 50))
        rec = repo.get_recommendation(tmp_db, rec_id)
        if rec["decision"] != "propose":
            continue
        t = t + timedelta(seconds=1)
        review.approve(tmp_db, rec_id, actor="bot", note=None, now=t, hard_cap_bp=5000)
        approved += 1
    assert verify_chain(tmp_db) == []
    total = sum(len(repo.changes_for_sku(tmp_db, s)) for s in skus)
    assert total == 100
    assert len(history(tmp_db, skus[0])) == len(repo.changes_for_sku(tmp_db, skus[0]))

    # Tamper via raw SQL with the trigger temporarily dropped (simulates a DBA edit).
    victim = repo.changes_for_sku(tmp_db, "S2")[3]
    tmp_db.execute("DROP TRIGGER trg_price_changes_no_update")
    tmp_db.execute(
        "UPDATE price_changes SET new_price_cents = new_price_cents + 1 WHERE change_id = ?",
        (victim["change_id"],),
    )
    breaks = verify_chain(tmp_db)
    assert breaks, "tamper not detected"
    assert any(b.sku == "S2" and b.change_id == victim["change_id"] for b in breaks)
    assert all(b.sku == "S2" for b in breaks)


def test_chain_detects_pointer_break(tmp_db: sqlite3.Connection, frozen_now: datetime) -> None:
    add_inventory(tmp_db, "S1", frozen_now)
    r1 = recommend(tmp_db, "S1", frozen_now)
    review.approve(tmp_db, r1, actor="a", note=None, now=frozen_now, hard_cap_bp=5000)
    r2 = recommend(tmp_db, "S1", frozen_now, competitor_prices=(4300, 4350))
    review.approve(
        tmp_db, r2, actor="a", note=None, now=frozen_now + timedelta(seconds=1), hard_cap_bp=5000
    )
    assert verify_chain(tmp_db) == []
    tmp_db.execute("DROP TRIGGER trg_price_changes_no_delete")
    first = repo.changes_for_sku(tmp_db, "S1")[0]
    tmp_db.execute("PRAGMA foreign_keys = OFF")
    tmp_db.execute("DELETE FROM price_changes WHERE change_id = ?", (first["change_id"],))
    breaks = verify_chain(tmp_db)
    assert breaks and "prev_change_id" in breaks[0].reason
