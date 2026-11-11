from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from repricer.engine.evaluate import evaluate
from repricer.engine.policy import PolicyError, PolicyStore, load_policies
from repricer.models import (
    Decision,
    EligibleOffer,
    InventorySnapshot,
    Policy,
    ReasonCode,
    Strategy,
)

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)

DEFAULT = Policy(
    strategy=Strategy.UNDERCUT_LOWEST,
    undercut_cents=50,
    undercut_bp=100,
    max_decrease_bp=1000,
    max_increase_bp=500,
    min_competitor_confidence=60,
    max_offer_age_hours=72,
    low_stock_threshold=3,
    round_to_cents=99,
)


def inv(**kw: object) -> InventorySnapshot:
    base = dict(
        sku="SKU-0001",
        title="t",
        cost_price_cents=3000,
        min_margin_bp=2000,
        current_price_cents=4500,
        stock_count=25,
        price_class="default",
    )
    base.update(kw)
    return InventorySnapshot(**base)  # type: ignore[arg-type]


def offer(price: int, conf: int = 85, oid: int = 1, age_h: int = 2, **kw: object) -> EligibleOffer:
    return EligibleOffer(
        offer_id=oid,
        competitor="DealHub",
        effective_price_cents=price,
        confidence=conf,
        scraped_at=NOW - timedelta(hours=age_h),
        **kw,  # type: ignore[arg-type]
    )


def test_ac_r2_worked_example() -> None:
    # cost 30, margin 20%, current 45, competitor 42.99 - $5 coupon = 37.99
    rec = evaluate(inv(), [offer(3799), offer(4449, oid=2)], DEFAULT, NOW)
    assert rec.floor_price_cents == 3600
    assert rec.lowest_competitor_cents == 3799
    assert rec.raw_target_cents == 3749
    assert rec.proposed_price_cents == 4099
    assert rec.decision is Decision.PROPOSE
    assert rec.reason_codes == (
        ReasonCode.UNDERCUT,
        ReasonCode.CLAMP_MAX_DECREASE,
        ReasonCode.ROUNDED,
    )
    assert rec.delta_cents == -401
    assert rec.competitor_offer_ids == (1, 2)
    assert "40.99" in rec.explanation


def test_ac_r3_floor_above_current_blocked() -> None:
    rec = evaluate(inv(current_price_cents=3500), [offer(3799)], DEFAULT, NOW)
    assert rec.decision is Decision.BLOCKED
    assert rec.reason_codes == (ReasonCode.FLOOR_ABOVE_CURRENT,)
    assert rec.proposed_price_cents == 3500
    assert rec.delta_cents == 0


def test_ac_r4_low_stock_holds() -> None:
    rec = evaluate(inv(stock_count=2), [offer(3799), offer(4449, oid=2)], DEFAULT, NOW)
    assert rec.decision is Decision.HOLD
    assert ReasonCode.LOW_STOCK_NO_DECREASE in rec.reason_codes
    assert rec.proposed_price_cents == 4500
    assert ReasonCode.ROUNDED not in rec.reason_codes


def test_low_stock_still_allows_increase() -> None:
    rec = evaluate(inv(stock_count=0), [offer(5000), offer(5100, oid=2)], DEFAULT, NOW)
    assert rec.decision is Decision.PROPOSE
    assert rec.proposed_price_cents > 4500
    assert ReasonCode.CLAMP_MAX_INCREASE in rec.reason_codes


def test_no_offers_holds() -> None:
    rec = evaluate(inv(), [], DEFAULT, NOW)
    assert rec.decision is Decision.HOLD
    assert rec.reason_codes == (ReasonCode.NO_COMPETITOR_DATA, ReasonCode.NO_CHANGE)
    assert rec.lowest_competitor_cents is None


def test_single_low_confidence_source_blocked() -> None:
    rec = evaluate(inv(), [offer(3799, conf=65)], DEFAULT, NOW)
    assert rec.decision is Decision.BLOCKED
    assert rec.reason_codes == (ReasonCode.SINGLE_SOURCE_LOW_CONF,)


def test_stale_and_low_confidence_offers_ignored() -> None:
    offers = [offer(1000, age_h=100), offer(2000, conf=10, oid=2), offer(4449, oid=3)]
    rec = evaluate(inv(), offers, DEFAULT, NOW)
    # only offer 3 is eligible -> single source but conf 85 -> fine
    assert rec.competitor_offer_ids == (3,)
    assert rec.lowest_competitor_cents == 4449


def test_floor_applied_when_competitor_below_cost() -> None:
    p = DEFAULT.model_copy(update={"max_decrease_bp": 5000})
    rec = evaluate(inv(), [offer(2000), offer(2100, oid=2)], p, NOW)
    assert rec.proposed_price_cents >= 3600
    assert ReasonCode.FLOOR_APPLIED in rec.reason_codes
    assert rec.proposed_price_cents == 3699  # floor 36.00 -> rounded up to .99, never down


def test_cost_anomaly_blocked() -> None:
    rec = evaluate(inv(cost_price_cents=100000, current_price_cents=4500), [], DEFAULT, NOW)
    assert rec.decision is Decision.BLOCKED
    assert rec.reason_codes == (ReasonCode.COST_ANOMALY,)


def test_hard_cap_blocks() -> None:
    p = DEFAULT.model_copy(update={"max_increase_bp": 9000, "round_to_cents": 0})
    rec = evaluate(inv(), [offer(9000), offer(9100, oid=2)], p, NOW, hard_cap_bp=5000)
    assert rec.decision is Decision.BLOCKED
    assert rec.reason_codes[-1] is ReasonCode.DELTA_EXCEEDS_HARD_CAP
    assert rec.proposed_price_cents == 4500


def test_target_margin_ignores_competitors() -> None:
    p = DEFAULT.model_copy(
        update={"strategy": Strategy.TARGET_MARGIN, "target_margin_bp": 4000, "round_to_cents": 0}
    )
    rec = evaluate(inv(), [offer(1000), offer(1100, oid=2)], p, NOW)
    assert rec.raw_target_cents == 4200
    assert rec.proposed_price_cents == 4200
    assert ReasonCode.TARGET_MARGIN in rec.reason_codes


def test_hold_strategy() -> None:
    p = DEFAULT.model_copy(update={"strategy": Strategy.HOLD})
    rec = evaluate(inv(), [offer(3799), offer(3899, oid=2)], p, NOW)
    assert rec.decision is Decision.HOLD
    assert rec.reason_codes == (ReasonCode.HOLD_STRATEGY, ReasonCode.NO_CHANGE)


def test_shipping_unknown_flagged() -> None:
    rec = evaluate(inv(), [offer(3799, shipping_unknown=True), offer(4449, oid=2)], DEFAULT, NOW)
    assert ReasonCode.SHIPPING_UNKNOWN in rec.reason_codes


# --- policy loading ---------------------------------------------------------


def test_bundled_policies_load() -> None:
    from repricer.config import DEFAULT_POLICIES_PATH

    ps = load_policies(DEFAULT_POLICIES_PATH)
    assert {"default", "clearance", "premium"} <= set(ps)


def test_policy_store_reload_keeps_old_on_error(tmp_path: Path) -> None:
    good = "default:\n  strategy: hold\n  max_decrease_bp: 0\n  max_increase_bp: 0\n"
    good += "  min_competitor_confidence: 50\n  max_offer_age_hours: 24\n"
    f = tmp_path / "p.yaml"
    f.write_text(good)
    store = PolicyStore(f)
    assert store.version == 1
    f.write_text("default:\n  strategy: nonsense\n")
    with pytest.raises(PolicyError):
        store.reload()
    assert store.version == 1
    assert store.for_class("default").strategy is Strategy.HOLD
    assert store.for_class("missing").strategy is Strategy.HOLD
    f.write_text(good.replace("hold", "match_lowest"))
    assert store.reload() == 2
    assert store.for_class("default").strategy is Strategy.MATCH_LOWEST


def test_policy_rejects_unknown_field(tmp_path: Path) -> None:
    f = tmp_path / "p.yaml"
    f.write_text(
        "default:\n  strategy: hold\n  max_decrease_bp: 0\n  max_increase_bp: 0\n"
        "  min_competitor_confidence: 50\n  max_offer_age_hours: 24\n  auto_apply: true\n"
    )
    with pytest.raises(PolicyError, match="auto_apply"):
        load_policies(f)
