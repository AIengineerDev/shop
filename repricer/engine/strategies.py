"""Step 3 of evaluation: raw target price before any bounds."""

from __future__ import annotations

from repricer.models import InventorySnapshot, Policy, ReasonCode, Strategy
from repricer.money import Cents, ceil_cents, fraction_bp


def raw_target(
    inv: InventorySnapshot, lowest: int | None, policy: Policy
) -> tuple[Cents, list[ReasonCode]]:
    current = Cents(inv.current_price_cents)
    if policy.strategy is Strategy.HOLD:
        return current, [ReasonCode.HOLD_STRATEGY]
    if policy.strategy is Strategy.TARGET_MARGIN:
        return ceil_cents(inv.cost_price_cents, policy.target_margin_bp), [ReasonCode.TARGET_MARGIN]
    if lowest is None:
        return current, [ReasonCode.NO_COMPETITOR_DATA]
    if policy.strategy is Strategy.MATCH_LOWEST:
        return Cents(lowest), [ReasonCode.MATCH_LOWEST]
    # UNDERCUT_LOWEST: subtract the larger of the fixed and the proportional undercut.
    cut = max(policy.undercut_cents, fraction_bp(lowest, policy.undercut_bp))
    return Cents(max(lowest - cut, 0)), [ReasonCode.UNDERCUT]
