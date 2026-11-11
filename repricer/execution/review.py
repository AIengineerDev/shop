"""Review state machine (spec §6.3).

pending --approve--> approved --apply--> applied
pending --reject---> rejected
pending --override-> approved (with price, re-validated) --apply--> applied
pending|approved --(new run / stale)--> expired
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from repricer.db import repo
from repricer.db.connection import transaction
from repricer.engine import guardrails
from repricer.execution.apply import apply_change_locked
from repricer.execution.errors import (
    FloorViolation,
    GuardrailViolation,
    IllegalTransition,
    StaleSnapshot,
)
from repricer.models import ChangeSource, Decision, ReasonCode, ReviewAction, ReviewStatus
from repricer.money import fmt

TRANSITIONS: dict[tuple[ReviewStatus, str], ReviewStatus] = {
    (ReviewStatus.PENDING, ReviewAction.APPROVE.value): ReviewStatus.APPROVED,
    (ReviewStatus.PENDING, ReviewAction.REJECT.value): ReviewStatus.REJECTED,
    (ReviewStatus.PENDING, ReviewAction.OVERRIDE.value): ReviewStatus.APPROVED,
    (ReviewStatus.PENDING, "expire"): ReviewStatus.EXPIRED,
    (ReviewStatus.APPROVED, "expire"): ReviewStatus.EXPIRED,
    (ReviewStatus.APPROVED, "apply"): ReviewStatus.APPLIED,
}


def transition(current: ReviewStatus, action: str) -> ReviewStatus:
    try:
        return TRANSITIONS[(current, action)]
    except KeyError:
        raise IllegalTransition(f"cannot {action} a {current.value} recommendation") from None


def _load_pending(conn: sqlite3.Connection, rec_id: str, action: str) -> sqlite3.Row:
    rec = repo.get_recommendation(conn, rec_id)
    transition(ReviewStatus(rec["review_status"]), action)  # raises if illegal
    return rec


def approve(
    conn: sqlite3.Connection,
    rec_id: str,
    *,
    actor: str,
    note: str | None,
    now: datetime,
    hard_cap_bp: int,
) -> str:
    """Approve and apply immediately. Returns change_id. Raises on illegal/stale."""
    try:
        with transaction(conn):
            rec = _load_pending(conn, rec_id, ReviewAction.APPROVE.value)
            if rec["decision"] != Decision.PROPOSE.value:
                raise IllegalTransition(
                    f"recommendation {rec_id} is '{rec['decision']}'; only 'propose' can be "
                    "approved (use override to set a price by hand)"
                )
            action_id = repo.insert_review_action(
                conn,
                recommendation_id=rec_id,
                action=ReviewAction.APPROVE.value,
                actor=actor,
                override_price_cents=None,
                note=note,
                now=now,
            )
            repo.set_review_status(conn, rec_id, ReviewStatus.PENDING, ReviewStatus.APPROVED)
            return apply_change_locked(
                conn,
                recommendation_id=rec_id,
                action_id=action_id,
                actor=actor,
                source=ChangeSource.REVIEW_APPROVE,
                now=now,
                hard_cap_bp=hard_cap_bp,
            )
    except StaleSnapshot:
        _expire(conn, rec_id)
        raise


def reject(
    conn: sqlite3.Connection, rec_id: str, *, actor: str, note: str | None, now: datetime
) -> str:
    with transaction(conn):
        _load_pending(conn, rec_id, ReviewAction.REJECT.value)
        action_id = repo.insert_review_action(
            conn,
            recommendation_id=rec_id,
            action=ReviewAction.REJECT.value,
            actor=actor,
            override_price_cents=None,
            note=note,
            now=now,
        )
        repo.set_review_status(conn, rec_id, ReviewStatus.PENDING, ReviewStatus.REJECTED)
        return action_id


def override(
    conn: sqlite3.Connection,
    rec_id: str,
    *,
    actor: str,
    price_cents: int,
    note: str | None,
    now: datetime,
    hard_cap_bp: int,
) -> str:
    """Reviewer sets their own price. Re-validated against floor, range and hard cap."""
    try:
        with transaction(conn):
            rec = _load_pending(conn, rec_id, ReviewAction.OVERRIDE.value)
            inv = repo.get_inventory(conn, rec["sku"])
            live = guardrails.check_live(inv, price_cents, hard_cap_bp)
            if live is ReasonCode.FLOOR_APPLIED:
                raise FloorViolation(
                    f"override {fmt(price_cents)} is below margin floor "
                    f"{fmt(rec['floor_price_cents'])} for {inv.sku}"
                )
            if live is not None:
                raise GuardrailViolation(live, f"{live.value}: override {fmt(price_cents)}")
            if price_cents == inv.current_price_cents:
                raise IllegalTransition("override price equals current price; reject instead")
            action_id = repo.insert_review_action(
                conn,
                recommendation_id=rec_id,
                action=ReviewAction.OVERRIDE.value,
                actor=actor,
                override_price_cents=price_cents,
                note=note,
                now=now,
            )
            repo.set_review_status(conn, rec_id, ReviewStatus.PENDING, ReviewStatus.APPROVED)
            return apply_change_locked(
                conn,
                recommendation_id=rec_id,
                action_id=action_id,
                actor=actor,
                source=ChangeSource.OVERRIDE,
                now=now,
                hard_cap_bp=hard_cap_bp,
                price_cents=price_cents,
            )
    except StaleSnapshot:
        _expire(conn, rec_id)
        raise


def _expire(conn: sqlite3.Connection, rec_id: str) -> None:
    """Mark a recommendation expired in its own transaction (after a rollback)."""
    with transaction(conn):
        rec = repo.get_recommendation(conn, rec_id)
        status = ReviewStatus(rec["review_status"])
        if (status, "expire") in TRANSITIONS:
            repo.set_review_status(conn, rec_id, status, ReviewStatus.EXPIRED)
