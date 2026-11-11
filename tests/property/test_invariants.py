"""Spec §5.6 invariants, property-based."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from repricer.engine.evaluate import evaluate
from repricer.models import (
    Decision,
    EligibleOffer,
    InventorySnapshot,
    Policy,
    ReasonCode,
    Strategy,
)
from repricer.money import apply_bp

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)

inventories = st.builds(
    InventorySnapshot,
    sku=st.just("S"),
    title=st.just("t"),
    cost_price_cents=st.integers(1, 2_000_000),
    min_margin_bp=st.integers(0, 10_000),
    current_price_cents=st.integers(1, 5_000_000),
    stock_count=st.integers(0, 500),
    price_class=st.just("default"),
)

offers = st.lists(
    st.builds(
        EligibleOffer,
        offer_id=st.integers(1, 10_000),
        competitor=st.just("c"),
        effective_price_cents=st.integers(0, 6_000_000),
        confidence=st.integers(0, 100),
        scraped_at=st.integers(0, 200).map(lambda h: NOW - timedelta(hours=h)),
        shipping_unknown=st.booleans(),
    ),
    max_size=6,
)

policies = st.builds(
    Policy,
    strategy=st.sampled_from(list(Strategy)),
    undercut_cents=st.integers(0, 500),
    undercut_bp=st.integers(0, 2000),
    target_margin_bp=st.integers(0, 20_000),
    max_decrease_bp=st.integers(0, 10_000),
    max_increase_bp=st.integers(0, 20_000),
    min_competitor_confidence=st.integers(0, 100),
    max_offer_age_hours=st.integers(1, 300),
    low_stock_threshold=st.integers(0, 10),
    round_to_cents=st.integers(0, 99),
)

hard_caps = st.integers(0, 20_000)


@settings(max_examples=2000, deadline=None)
@given(inventories, offers, policies, hard_caps)
def test_proposed_never_below_floor_or_cost(
    inv: InventorySnapshot, offs: list[EligibleOffer], policy: Policy, cap: int
) -> None:
    rec = evaluate(inv, offs, policy, NOW, hard_cap_bp=cap)
    if rec.decision is Decision.BLOCKED:
        assert rec.proposed_price_cents == inv.current_price_cents
        return
    assert rec.proposed_price_cents >= rec.floor_price_cents
    assert rec.proposed_price_cents >= inv.cost_price_cents


@settings(max_examples=2000, deadline=None)
@given(inventories, offers, policies, hard_caps)
def test_bounded_delta(
    inv: InventorySnapshot, offs: list[EligibleOffer], policy: Policy, cap: int
) -> None:
    rec = evaluate(inv, offs, policy, NOW, hard_cap_bp=cap)
    if rec.decision is Decision.BLOCKED:
        return
    lo = apply_bp(inv.current_price_cents, -policy.max_decrease_bp)
    hi = apply_bp(inv.current_price_cents, policy.max_increase_bp)
    # Never below the lower bound (rounding is floored at it); at most 99c of rounding
    # slack above the upper bound; floor may lift above both.
    if ReasonCode.FLOOR_APPLIED not in rec.reason_codes:
        assert rec.proposed_price_cents >= lo
        assert rec.proposed_price_cents <= hi + 99
    assert abs(rec.delta_bp) <= cap


@settings(max_examples=2000, deadline=None)
@given(inventories, offers, policies, hard_caps)
def test_deterministic_and_consistent(
    inv: InventorySnapshot, offs: list[EligibleOffer], policy: Policy, cap: int
) -> None:
    a = evaluate(inv, offs, policy, NOW, hard_cap_bp=cap)
    b = evaluate(inv, offs, policy, NOW, hard_cap_bp=cap)
    assert a == b
    assert a.delta_cents == a.proposed_price_cents - a.current_price_cents
    assert (a.decision is Decision.PROPOSE) == (a.proposed_price_cents != a.current_price_cents)
    assert a.explanation
    assert len(a.reason_codes) >= 1
