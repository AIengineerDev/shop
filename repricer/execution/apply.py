"""The one path that changes a live price (spec §6.2).

``inventory.current_price_cents`` is written only here, in the same transaction that
appends the ``price_changes`` row. Nothing else may call ``repo._set_inventory_price``.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from repricer.clock import iso
from repricer.db import repo
from repricer.db.connection import transaction
from repricer.engine import guardrails
from repricer.execution.errors import FloorViolation, GuardrailViolation, StaleSnapshot
from repricer.execution.hashing import row_hash
from repricer.ids import new_ulid
from repricer.models import ChangeSource, ReasonCode, ReviewStatus
from repricer.money import fmt


def apply_change(
    conn: sqlite3.Connection,
    *,
    recommendation_id: str,
    action_id: str,
    actor: str,
    source: ChangeSource,
    now: datetime,
    hard_cap_bp: int,
    price_cents: int | None = None,
) -> str:
    """Apply an *approved* recommendation. Returns the new change_id.

    ``price_cents`` overrides the recommendation's proposed price (override flow); it has
    already been validated by the caller but is re-validated here against the live row.
    """
    with transaction(conn):
        return apply_change_locked(
            conn,
            recommendation_id=recommendation_id,
            action_id=action_id,
            actor=actor,
            source=source,
            now=now,
            hard_cap_bp=hard_cap_bp,
            price_cents=price_cents,
        )


def apply_change_locked(
    conn: sqlite3.Connection,
    *,
    recommendation_id: str,
    action_id: str,
    actor: str,
    source: ChangeSource,
    now: datetime,
    hard_cap_bp: int,
    price_cents: int | None = None,
) -> str:
    """Same as ``apply_change`` but assumes the caller holds an IMMEDIATE transaction."""
    rec = repo.get_recommendation(conn, recommendation_id)
    if rec["review_status"] != ReviewStatus.APPROVED.value:
        raise StaleSnapshot(
            f"recommendation {recommendation_id} is {rec['review_status']}, not approved"
        )
    inv = repo.get_inventory(conn, rec["sku"])

    if inv.current_price_cents != rec["current_price_cents"]:
        # Snapshot no longer matches the live price. The caller's transaction rolls back
        # and review.py marks the recommendation expired in a follow-up transaction.
        raise StaleSnapshot(
            f"live price {fmt(inv.current_price_cents)} != snapshot "
            f"{fmt(rec['current_price_cents'])} for {inv.sku}"
        )

    new_price = rec["proposed_price_cents"] if price_cents is None else price_cents
    live = guardrails.check_live(inv, new_price, hard_cap_bp)
    if live is ReasonCode.FLOOR_APPLIED:
        raise FloorViolation(f"price {fmt(new_price)} is below margin floor for {inv.sku}")
    if live is not None:
        raise GuardrailViolation(live, f"{live.value} for {inv.sku} at {fmt(new_price)}")

    prev = repo.last_change_for_sku(conn, inv.sku)
    row = {
        "change_id": new_ulid(),
        "sku": inv.sku,
        "recommendation_id": recommendation_id,
        "action_id": action_id,
        "old_price_cents": inv.current_price_cents,
        "new_price_cents": new_price,
        "source": source.value,
        "actor": actor,
        "applied_at": iso(now),
        "prev_change_id": None if prev is None else prev["change_id"],
    }
    row["row_hash"] = row_hash(row)
    repo.insert_change(conn, row)
    repo._set_inventory_price(conn, inv.sku, inv.current_price_cents, new_price, now)
    repo.set_review_status(conn, recommendation_id, ReviewStatus.APPROVED, ReviewStatus.APPLIED)
    return str(row["change_id"])
