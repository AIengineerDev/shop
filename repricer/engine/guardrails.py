"""Hard guardrails (spec §5.4). Any hit -> decision 'blocked', never proposed."""

from __future__ import annotations

from repricer.models import EligibleOffer, InventorySnapshot, ReasonCode
from repricer.money import delta_bp

SINGLE_SOURCE_MIN_CONF = 70
MIN_PRICE_CENTS = 1
MAX_PRICE_CENTS = 100_000_000  # $1,000,000
COST_ANOMALY_MULTIPLE = 10


def pre_checks(inv: InventorySnapshot, floor: int) -> ReasonCode | None:
    """Run before any strategy: anomalies in the inventory row itself."""
    if inv.cost_price_cents <= 0 or inv.cost_price_cents > inv.current_price_cents * (
        COST_ANOMALY_MULTIPLE
    ):
        return ReasonCode.COST_ANOMALY
    if floor > inv.current_price_cents:
        return ReasonCode.FLOOR_ABOVE_CURRENT
    return None


def offer_checks(offers: list[EligibleOffer]) -> ReasonCode | None:
    if len(offers) == 1 and offers[0].confidence < SINGLE_SOURCE_MIN_CONF:
        return ReasonCode.SINGLE_SOURCE_LOW_CONF
    return None


def post_checks(current: int, proposed: int, hard_cap_bp: int) -> ReasonCode | None:
    """Run on the final proposed price; also reused for live re-validation on apply."""
    if proposed < MIN_PRICE_CENTS or proposed > MAX_PRICE_CENTS:
        return ReasonCode.PRICE_OUT_OF_RANGE
    if abs(delta_bp(current, proposed)) > hard_cap_bp:
        return ReasonCode.DELTA_EXCEEDS_HARD_CAP
    return None


def check_live(inv: InventorySnapshot, proposed: int, hard_cap_bp: int) -> ReasonCode | None:
    """Re-validate a price against the *live* inventory row at apply/override time."""
    from repricer.money import ceil_cents  # local import keeps module deps acyclic

    floor = ceil_cents(inv.cost_price_cents, inv.min_margin_bp)
    pre = pre_checks(inv, floor)
    if pre is ReasonCode.COST_ANOMALY:
        return pre
    if proposed < floor:
        return ReasonCode.FLOOR_APPLIED  # caller maps this to a floor violation
    return post_checks(inv.current_price_cents, proposed, hard_cap_bp)
