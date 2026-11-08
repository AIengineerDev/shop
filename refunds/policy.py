"""The pure decision.

`decide()` takes data in and returns a Decision. No clock, no I/O, no storage: `now`, the ledger
snapshot and external signals are all parameters, so every rule is unit-testable with literals.

Rules are small functions `Context -> Optional[Verdict]`, evaluated in order. A rule may:
  * return None            -> nothing to say
  * REJECT                 -> the request is malformed/impossible; evaluation stops (later rules
                              could not even be evaluated meaningfully, e.g. currency mismatch)
  * ESCALATE               -> recorded with its reason; evaluation continues so the reviewer sees
                              *every* failing reason, not just the first
  * APPROVE with an amount -> adjusts the effective amount (e.g. cap to what is still refundable);
                              later rules see the adjusted amount
Severity: REJECT > ESCALATE > APPROVE.

NOTE: the over-refund invariant is *not* enforced here. `cap_to_remaining` is a courtesy so the
customer gets a sensible partial refund; the ledger is the single place that guarantees the money
can never exceed what was paid.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Sequence

from refunds.config import PolicyConfig
from refunds.model import (
    REFUNDABLE_STATUSES,
    Decision,
    LedgerView,
    Order,
    OrderStatus,
    Outcome,
    RefundReason,
    RefundRequest,
    Signals,
)
from refunds.money import Money


@dataclass(frozen=True)
class Context:
    order: Order
    request: RefundRequest
    ledger: LedgerView
    now: datetime
    signals: Signals
    config: PolicyConfig
    amount: Money  # effective amount, threaded through amount-adjusting rules


@dataclass(frozen=True)
class Verdict:
    outcome: Outcome
    reason: str
    amount: Optional[Money] = None  # only meaningful with Outcome.APPROVE


Rule = Callable[[Context], Optional[Verdict]]


# ------------------------------------------------------------------ rules
def order_is_refundable(ctx: Context) -> Optional[Verdict]:
    if ctx.order.status not in REFUNDABLE_STATUSES:
        return Verdict(Outcome.REJECT, f"order_not_refundable:status={ctx.order.status.value}")
    return None


def amount_is_valid(ctx: Context) -> Optional[Verdict]:
    if ctx.amount.currency != ctx.order.paid_total.currency:
        return Verdict(Outcome.REJECT, f"currency_mismatch:{ctx.amount.currency}!={ctx.order.paid_total.currency}")
    if not ctx.amount.is_positive:
        return Verdict(Outcome.REJECT, f"amount_invalid:{ctx.amount}")
    return None


def cap_to_remaining(ctx: Context) -> Optional[Verdict]:
    remaining = ctx.ledger.remaining
    if not remaining.is_positive:
        return Verdict(Outcome.REJECT, "nothing_left_to_refund")
    if ctx.amount > remaining:
        return Verdict(Outcome.APPROVE, f"capped_to_remaining:{remaining}", amount=remaining)
    return None


def within_window(ctx: Context) -> Optional[Verdict]:
    anchor = ctx.order.delivered_at or ctx.order.paid_at
    limit = anchor + timedelta(days=ctx.config.window_days)
    if ctx.now > limit:
        return Verdict(Outcome.ESCALATE, f"outside_window:{ctx.config.window_days}d")
    return None


def reason_is_automatic(ctx: Context) -> Optional[Verdict]:
    r = ctx.request.reason
    if r is RefundReason.OTHER:
        return Verdict(Outcome.ESCALATE, "reason_requires_review:other")
    if r is RefundReason.CHANGED_MIND and ctx.order.status is not OrderStatus.PAID:
        return Verdict(Outcome.ESCALATE, "reason_requires_review:changed_mind_after_shipment")
    return None


def under_auto_limit(ctx: Context) -> Optional[Verdict]:
    limit = Money(ctx.config.auto_limit, ctx.order.paid_total.currency)
    if ctx.amount > limit:
        return Verdict(Outcome.ESCALATE, f"above_auto_limit:{limit}")
    return None


def history_under_threshold(ctx: Context) -> Optional[Verdict]:
    n, mx = ctx.signals.customer_refunds_in_window, ctx.config.history_max_refunds
    if n >= mx:
        return Verdict(Outcome.ESCALATE, f"history_threshold:{n}>={mx}_in_{ctx.config.history_window_days}d")
    return None


DEFAULT_RULES: Sequence[Rule] = (
    order_is_refundable,
    amount_is_valid,
    cap_to_remaining,
    within_window,
    reason_is_automatic,
    under_auto_limit,
    history_under_threshold,
)

SEVERITY = {Outcome.REJECT: 2, Outcome.ESCALATE: 1, Outcome.APPROVE: 0}


# ------------------------------------------------------------------ evaluation
def decide(
    order: Order,
    request: RefundRequest,
    ledger: LedgerView,
    now: datetime,
    config: PolicyConfig,
    signals: Signals = Signals(),
    rules: Sequence[Rule] = DEFAULT_RULES,
) -> Decision:
    ctx = Context(order, request, ledger, now, signals, config, amount=request.amount)
    outcome = Outcome.APPROVE
    reasons: List[str] = []
    for rule in rules:
        v = rule(ctx)
        if v is None:
            continue
        reasons.append(v.reason)
        if v.outcome is Outcome.REJECT:
            return Decision(Outcome.REJECT, None, reasons, config.version)
        if v.outcome is Outcome.APPROVE and v.amount is not None:
            ctx = replace(ctx, amount=v.amount)
        elif SEVERITY[v.outcome] > SEVERITY[outcome]:
            outcome = v.outcome
    if outcome is Outcome.APPROVE:
        return Decision(outcome, ctx.amount, reasons or ["eligible"], config.version)
    return Decision(outcome, None, reasons, config.version)
