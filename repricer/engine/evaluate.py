"""Spec §5.2: the fixed evaluation order. Pure function of (inventory, offers, policy, now).

Steps:
 1. floor      = ceil(cost × (1 + min_margin))
 2. lowest     = min(effective price over eligible offers) or None
 3. raw_target = strategy(lowest, current, policy)
 4. bounded    = clamp(raw_target, current × (1 − max_decrease), current × (1 + max_increase))
 5. floored    = max(bounded, floor)
 6. stock      = if stock ≤ threshold and floored < current: floored = current
 7. rounded    = round_to(floored, ending, floor=max(floor, lower bound))  [skipped if == current]
 8. decision   = propose / hold / blocked
"""

from __future__ import annotations

from datetime import datetime, timedelta

from repricer.engine import guardrails
from repricer.engine.explain import explain
from repricer.engine.strategies import raw_target
from repricer.models import (
    Decision,
    EligibleOffer,
    InventorySnapshot,
    Policy,
    ReasonCode,
    Recommendation,
)
from repricer.money import Cents, apply_bp, ceil_cents, clamp, delta_bp, round_to


def filter_eligible(
    offers: list[EligibleOffer], policy: Policy, now: datetime
) -> list[EligibleOffer]:
    """Re-apply the policy-dependent eligibility rules in code (the DB query does too)."""
    cutoff = now - timedelta(hours=policy.max_offer_age_hours)
    return sorted(
        (
            o
            for o in offers
            if o.confidence >= policy.min_competitor_confidence and o.scraped_at >= cutoff
        ),
        key=lambda o: (o.effective_price_cents, o.offer_id),
    )


def evaluate(
    inv: InventorySnapshot,
    offers: list[EligibleOffer],
    policy: Policy,
    now: datetime,
    *,
    hard_cap_bp: int = 5000,
) -> Recommendation:
    current = Cents(inv.current_price_cents)
    codes: list[ReasonCode] = []
    eligible = filter_eligible(offers, policy, now)
    offer_ids = tuple(o.offer_id for o in eligible)

    # 1. floor
    floor = ceil_cents(inv.cost_price_cents, inv.min_margin_bp)

    # 2. lowest
    lowest: int | None = eligible[0].effective_price_cents if eligible else None

    ctx = {
        "cost": inv.cost_price_cents,
        "current": current,
        "floor": floor,
        "lowest": lowest,
        "stock": inv.stock_count,
    }

    def blocked(code: ReasonCode, raw: int) -> Recommendation:
        codes.append(code)
        ctx.update({"raw_target": raw, "bounded": current, "proposed": current, "delta_bp": 0})
        return Recommendation(
            sku=inv.sku,
            current_price_cents=current,
            cost_price_cents=inv.cost_price_cents,
            min_margin_bp=inv.min_margin_bp,
            floor_price_cents=floor,
            lowest_competitor_cents=lowest,
            competitor_offer_ids=offer_ids,
            raw_target_cents=raw,
            proposed_price_cents=current,
            delta_cents=0,
            delta_bp=0,
            decision=Decision.BLOCKED,
            reason_codes=tuple(codes),
            explanation=explain(codes, ctx),
        )

    pre = guardrails.pre_checks(inv, floor)
    if pre is not None:
        return blocked(pre, current)
    offer_guard = guardrails.offer_checks(eligible)
    if offer_guard is not None:
        return blocked(offer_guard, current)

    # 3. strategy
    raw, strategy_codes = raw_target(inv, lowest, policy)
    codes.extend(strategy_codes)
    if eligible and eligible[0].shipping_unknown and lowest is not None:
        codes.append(ReasonCode.SHIPPING_UNKNOWN)
    ctx["raw_target"] = raw

    # 4. bounds
    lo = apply_bp(current, -policy.max_decrease_bp)
    hi = apply_bp(current, policy.max_increase_bp)
    bounded = clamp(raw, lo, hi)
    if bounded != raw:
        codes.append(
            ReasonCode.CLAMP_MAX_DECREASE if raw < bounded else ReasonCode.CLAMP_MAX_INCREASE
        )
    ctx["bounded"] = bounded

    # 5. floor
    floored = Cents(max(bounded, floor))
    if floored != bounded:
        codes.append(ReasonCode.FLOOR_APPLIED)

    # 6. stock guard
    if inv.stock_count <= policy.low_stock_threshold and floored < current:
        floored = current
        codes.append(ReasonCode.LOW_STOCK_NO_DECREASE)

    # 7. rounding (only when the price actually moves; never below floor or lower bound)
    proposed = floored
    if floored != current:
        rounded = round_to(floored, policy.round_to_cents, max(floor, lo))
        if rounded != floored:
            codes.append(ReasonCode.ROUNDED)
        proposed = rounded
    ctx["proposed"] = proposed

    # 8. decision
    d_bp = delta_bp(current, proposed)
    ctx["delta_bp"] = d_bp
    post = guardrails.post_checks(current, proposed, hard_cap_bp)
    if post is not None:
        return blocked(post, raw)
    if proposed == current:
        codes.append(ReasonCode.NO_CHANGE)
        decision = Decision.HOLD
    else:
        decision = Decision.PROPOSE

    return Recommendation(
        sku=inv.sku,
        current_price_cents=current,
        cost_price_cents=inv.cost_price_cents,
        min_margin_bp=inv.min_margin_bp,
        floor_price_cents=floor,
        lowest_competitor_cents=lowest,
        competitor_offer_ids=offer_ids,
        raw_target_cents=raw,
        proposed_price_cents=proposed,
        delta_cents=proposed - current,
        delta_bp=d_bp,
        decision=decision,
        reason_codes=tuple(codes),
        explanation=explain(codes, ctx),
    )
