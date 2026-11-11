from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from repricer.db import repo
from repricer.execution import review
from repricer.execution.errors import FloorViolation, IllegalTransition
from repricer.models import ReviewStatus
from tests.unit.helpers import add_inventory, recommend


@pytest.mark.parametrize(
    "status,action,expected",
    [
        (ReviewStatus.PENDING, "approve", ReviewStatus.APPROVED),
        (ReviewStatus.PENDING, "reject", ReviewStatus.REJECTED),
        (ReviewStatus.PENDING, "override", ReviewStatus.APPROVED),
        (ReviewStatus.PENDING, "expire", ReviewStatus.EXPIRED),
        (ReviewStatus.APPROVED, "apply", ReviewStatus.APPLIED),
        (ReviewStatus.APPROVED, "expire", ReviewStatus.EXPIRED),
    ],
)
def test_legal_transitions(status: ReviewStatus, action: str, expected: ReviewStatus) -> None:
    assert review.transition(status, action) is expected


@pytest.mark.parametrize(
    "status,action",
    [
        (ReviewStatus.APPLIED, "approve"),
        (ReviewStatus.REJECTED, "approve"),
        (ReviewStatus.EXPIRED, "approve"),
        (ReviewStatus.APPROVED, "approve"),
        (ReviewStatus.APPLIED, "reject"),
        (ReviewStatus.PENDING, "apply"),
        (ReviewStatus.REJECTED, "override"),
    ],
)
def test_illegal_transitions(status: ReviewStatus, action: str) -> None:
    with pytest.raises(IllegalTransition):
        review.transition(status, action)


def test_approve_applies_and_records(tmp_db: sqlite3.Connection, frozen_now: datetime) -> None:
    add_inventory(tmp_db, "S1", frozen_now)
    rec_id = recommend(tmp_db, "S1", frozen_now)
    change_id = review.approve(
        tmp_db, rec_id, actor="alex", note="ok", now=frozen_now, hard_cap_bp=5000
    )
    assert repo.get_recommendation(tmp_db, rec_id)["review_status"] == "applied"
    assert repo.get_inventory(tmp_db, "S1").current_price_cents == 4099
    changes = repo.changes_for_sku(tmp_db, "S1")
    assert len(changes) == 1 and changes[0]["change_id"] == change_id
    assert changes[0]["source"] == "review_approve"
    actions = repo.actions_for_recommendation(tmp_db, rec_id)
    assert [a["action"] for a in actions] == ["approve"]
    # second approve is illegal
    with pytest.raises(IllegalTransition):
        review.approve(tmp_db, rec_id, actor="alex", note=None, now=frozen_now, hard_cap_bp=5000)


def test_reject(tmp_db: sqlite3.Connection, frozen_now: datetime) -> None:
    add_inventory(tmp_db, "S1", frozen_now)
    rec_id = recommend(tmp_db, "S1", frozen_now)
    review.reject(tmp_db, rec_id, actor="alex", note="no", now=frozen_now)
    assert repo.get_recommendation(tmp_db, rec_id)["review_status"] == "rejected"
    assert repo.get_inventory(tmp_db, "S1").current_price_cents == 4500
    assert repo.changes_for_sku(tmp_db, "S1") == []


def test_ac_x5_hold_and_blocked_cannot_be_approved(
    tmp_db: sqlite3.Connection, frozen_now: datetime
) -> None:
    add_inventory(tmp_db, "HOLD", frozen_now)
    hold_id = recommend(tmp_db, "HOLD", frozen_now, competitor_prices=())
    assert repo.get_recommendation(tmp_db, hold_id)["decision"] == "hold"
    with pytest.raises(IllegalTransition):
        review.approve(tmp_db, hold_id, actor="a", note=None, now=frozen_now, hard_cap_bp=5000)
    add_inventory(tmp_db, "BLK", frozen_now, current=3500)
    blk_id = recommend(tmp_db, "BLK", frozen_now)
    assert repo.get_recommendation(tmp_db, blk_id)["decision"] == "blocked"
    with pytest.raises(IllegalTransition):
        review.approve(tmp_db, blk_id, actor="a", note=None, now=frozen_now, hard_cap_bp=5000)
    # still pending, nothing recorded
    assert repo.get_recommendation(tmp_db, blk_id)["review_status"] == "pending"
    assert repo.actions_for_recommendation(tmp_db, blk_id) == []


def test_override_below_floor_rejected(tmp_db: sqlite3.Connection, frozen_now: datetime) -> None:
    add_inventory(tmp_db, "S1", frozen_now)
    rec_id = recommend(tmp_db, "S1", frozen_now)
    with pytest.raises(FloorViolation):
        review.override(
            tmp_db, rec_id, actor="a", price_cents=3599, note=None, now=frozen_now, hard_cap_bp=5000
        )
    assert repo.get_recommendation(tmp_db, rec_id)["review_status"] == "pending"
    assert repo.actions_for_recommendation(tmp_db, rec_id) == []


def test_override_within_bounds_applies(tmp_db: sqlite3.Connection, frozen_now: datetime) -> None:
    add_inventory(tmp_db, "S1", frozen_now)
    rec_id = recommend(tmp_db, "S1", frozen_now)
    review.override(
        tmp_db, rec_id, actor="a", price_cents=4199, note="meet", now=frozen_now, hard_cap_bp=5000
    )
    assert repo.get_inventory(tmp_db, "S1").current_price_cents == 4199
    change = repo.changes_for_sku(tmp_db, "S1")[0]
    assert change["source"] == "override"
    act = repo.actions_for_recommendation(tmp_db, rec_id)[0]
    assert act["action"] == "override" and act["override_price_cents"] == 4199


def test_override_can_rescue_blocked(tmp_db: sqlite3.Connection, frozen_now: datetime) -> None:
    # FLOOR_ABOVE_CURRENT: current 35 < floor 36. Human sets 36.50.
    add_inventory(tmp_db, "BLK", frozen_now, current=3500)
    rec_id = recommend(tmp_db, "BLK", frozen_now)
    review.override(
        tmp_db, rec_id, actor="a", price_cents=3650, note=None, now=frozen_now, hard_cap_bp=5000
    )
    assert repo.get_inventory(tmp_db, "BLK").current_price_cents == 3650
