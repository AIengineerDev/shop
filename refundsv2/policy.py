"""Policy as data + a pure decision function.

`PolicyConfig` holds every threshold and is versioned; `RULES` is an ordered registry of small
functions `Context -> RuleResult`. `decide()` folds the results:

  * `reject`   stops immediately: the request is impossible (wrong order state, bad amount, nothing left)
  * `cap`      lowers the effective amount (what is left on the order)
  * `limit`    lowers the automatic-approval ceiling (abuse score)
  * `escalate` marks the request for a person; all escalate reasons are collected, nothing stops

Adding a rule is one function and one line in RULES. Nothing here touches storage or the clock:
the service passes `now`, the ledger view and the signals in.

Note what this module does NOT do: it does not guarantee the over-refund invariant. `cap_to_remaining`
produces a sensible partial amount, but the ledger's conditional insert is the only enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from refundsv2.model import (Decision, LedgerView, Order, OrderStatus, Outcome, Reason, RefundRequest, RuleResult, Signals)
from refundsv2.money import format_cents

DAY = 86400.0


class PolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    version: str
    refundable_statuses: Tuple[OrderStatus, ...] = (OrderStatus.PAID, OrderStatus.SHIPPED, OrderStatus.DELIVERED)
    window_days: int = Field(default=30, ge=0)
    auto_limit_cents: int = Field(default=20_000, ge=0)
    auto_reasons: Tuple[Reason, ...] = (Reason.DAMAGED, Reason.NOT_AS_DESCRIBED, Reason.LATE, Reason.NOT_RECEIVED)
    changed_mind_before_shipment: bool = True
    max_refunds_90d: int = Field(default=3, ge=0)
    abuse_escalate_at: float = Field(default=0.8, ge=0, le=1)
    abuse_shrink_at: float = Field(default=0.5, ge=0, le=1)
    abuse_shrink_factor: float = Field(default=0.5, gt=0, le=1)


DEFAULT_POLICY = PolicyConfig(version="v1")


@dataclass(frozen=True)
class Context:
    order: Order
    request: RefundRequest
    ledger: LedgerView
    signals: Signals
    policy: PolicyConfig
    now: float
    amount_cents: int      # effective amount so far (after caps)
    limit_cents: int       # effective auto-approval ceiling so far


def _pass(name: str, reason: str = "") -> RuleResult:
    return RuleResult(name, True, reason)


def order_refundable(c: Context) -> RuleResult:
    if c.order.status in c.policy.refundable_statuses:
        return _pass("order_refundable")
    return RuleResult("order_refundable", False, f"order_not_refundable:status={c.order.status.value}", "reject")


def amount_valid(c: Context) -> RuleResult:
    if c.request.currency != c.order.currency:
        return RuleResult("amount_valid", False, f"currency_mismatch:{c.request.currency}!={c.order.currency}", "reject")
    if c.request.amount_cents <= 0:
        return RuleResult("amount_valid", False, "amount_invalid:must_be_positive", "reject")
    return _pass("amount_valid")


def cap_to_remaining(c: Context) -> RuleResult:
    remaining = c.ledger.remaining_cents
    if remaining <= 0:
        return RuleResult("cap_to_remaining", False, "nothing_left_to_refund", "reject")
    if c.amount_cents > remaining:
        return RuleResult("cap_to_remaining", True, f"capped_to_remaining:{format_cents(remaining)}", "cap", remaining)
    return _pass("cap_to_remaining")


def within_window(c: Context) -> RuleResult:
    anchor = c.order.delivered_at if c.order.delivered_at is not None else c.order.paid_at
    if anchor is None:
        return _pass("within_window", "no_anchor")
    if c.now - anchor <= c.policy.window_days * DAY:  # boundary inclusive
        return _pass("within_window")
    return RuleResult("within_window", False, f"outside_window:{c.policy.window_days}d", "escalate")


def reason_automatic(c: Context) -> RuleResult:
    r = c.request.reason
    if r in c.policy.auto_reasons:
        return _pass("reason_automatic")
    shipped = c.order.status in (OrderStatus.SHIPPED, OrderStatus.DELIVERED) or c.order.shipped_at is not None
    if r == Reason.CHANGED_MIND and c.policy.changed_mind_before_shipment and not shipped:
        return _pass("reason_automatic", "changed_mind_before_shipment")
    return RuleResult("reason_automatic", False, f"reason_requires_review:{r.value}", "escalate")


def history_threshold(c: Context) -> RuleResult:
    n, limit = c.signals.refunds_last_90d, c.policy.max_refunds_90d
    if n >= limit:
        return RuleResult("history_threshold", False, f"history_threshold:{n}>={limit}", "escalate")
    return _pass("history_threshold")


def abuse_score_bounded(c: Context) -> RuleResult:
    """The model can narrow the envelope, never widen it. No score -> no effect."""
    s = c.signals.abuse_score
    if s is None:
        return _pass("abuse_score_bounded", "abuse_score:absent")
    if s >= c.policy.abuse_escalate_at:
        return RuleResult("abuse_score_bounded", False, f"abuse_score:{s:.2f}>={c.policy.abuse_escalate_at:.2f}", "escalate")
    if s >= c.policy.abuse_shrink_at:
        new_limit = int(c.limit_cents * c.policy.abuse_shrink_factor)
        return RuleResult("abuse_score_bounded", True, f"abuse_score:{s:.2f}>={c.policy.abuse_shrink_at:.2f}:limit={format_cents(new_limit)}", "limit", new_limit)
    return _pass("abuse_score_bounded", f"abuse_score:{s:.2f}")


def under_auto_limit(c: Context) -> RuleResult:
    """Runs last: uses the capped amount and the possibly shrunk limit."""
    if c.amount_cents <= c.limit_cents:
        return _pass("under_auto_limit")
    return RuleResult("under_auto_limit", False, f"above_auto_limit:{format_cents(c.limit_cents)}", "escalate")


Rule = Callable[[Context], RuleResult]
RULES: List[Tuple[str, Rule]] = [
    ("order_refundable", order_refundable),
    ("amount_valid", amount_valid),
    ("cap_to_remaining", cap_to_remaining),
    ("within_window", within_window),
    ("reason_automatic", reason_automatic),
    ("history_threshold", history_threshold),
    ("abuse_score_bounded", abuse_score_bounded),
    ("under_auto_limit", under_auto_limit),
]


def decide(order: Order, request: RefundRequest, ledger: LedgerView, signals: Signals, policy: PolicyConfig, now: float,
           rules: Optional[List[Tuple[str, Rule]]] = None) -> Decision:
    rules = RULES if rules is None else rules
    ctx = Context(order, request, ledger, signals, policy, now, amount_cents=request.amount_cents, limit_cents=policy.auto_limit_cents)
    trace: List[RuleResult] = []
    reasons: List[str] = []
    escalate = False
    for _, rule in rules:
        res = rule(ctx)
        trace.append(res)
        if res.effect == "reject":
            return Decision(Outcome.REJECT, 0, (res.reason,), tuple(trace), policy.version)
        if res.effect == "cap":
            ctx = replace(ctx, amount_cents=res.cap_cents)
            reasons.append(res.reason)
        elif res.effect == "limit":
            ctx = replace(ctx, limit_cents=res.cap_cents)
            reasons.append(res.reason)
        elif res.effect == "escalate":
            escalate = True
            reasons.append(res.reason)
    outcome = Outcome.ESCALATE if escalate else Outcome.APPROVE
    if not escalate and not reasons:
        reasons.append("all_rules_passed")
    return Decision(outcome, ctx.amount_cents, tuple(reasons), tuple(trace), policy.version)
