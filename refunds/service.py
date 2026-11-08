"""Orchestration: idempotency -> load -> pure decide -> ledger reserve -> provider -> settle -> record.

Ordering matters for crash safety:
  1. the ledger reservation is written BEFORE the provider call (so a crash after the call still
     holds the money as reserved, never as free);
  2. the idempotency record is written LAST. If we crash before it, the retry re-decides, and the
     ledger returns the existing reservation for that request_id instead of a new one.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional

from refunds.config import PolicyConfig
from refunds.idempotency import IdempotencyStore
from refunds.ledger import ConcurrencyConflict, OverRefund, RefundLedger
from refunds.model import Decision, Order, Outcome, Refund, RefundStatus, RefundRequest, Signals
from refunds.policy import DEFAULT_RULES, Rule, decide
from refunds.provider import PaymentProvider, ProviderDeclined, ProviderTimeout


class OrderNotFound(Exception):
    pass


@dataclass(frozen=True)
class RefundResult:
    decision: Decision
    refund: Optional[Refund]


class OrderRepo:
    def __init__(self, orders: Iterable[Order] = ()) -> None:
        self._orders: Dict[str, Order] = {}
        for o in orders:
            self.add(o)

    def add(self, order: Order) -> None:
        self._orders[order.order_id] = order

    def get(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def for_customer(self, customer_id: str) -> List[Order]:
        return [o for o in self._orders.values() if o.customer_id == customer_id]


Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RefundService:
    def __init__(
        self,
        orders: OrderRepo,
        ledger: RefundLedger,
        provider: PaymentProvider,
        policy: PolicyConfig,
        idempotency: Optional[IdempotencyStore] = None,
        clock: Clock = utc_now,
        rules: Optional[List[Rule]] = None,
        abuse_scores: Optional[Callable[[str], Optional[float]]] = None,
    ) -> None:
        self.orders = orders
        self.ledger = ledger
        self.provider = provider
        self.policy = policy
        self.idempotency = idempotency or IdempotencyStore()
        self.clock = clock
        self.rules = list(rules) if rules is not None else list(DEFAULT_RULES)
        self.abuse_scores = abuse_scores or (lambda customer_id: None)
        self.max_attempts = 3
        for o in orders._orders.values():
            ledger.open_account(o.order_id, o.paid_total)

    # ---------------------------------------------------------------- main entry
    def handle(self, request: RefundRequest) -> RefundResult:
        fp = request.fingerprint()
        replay = self.idempotency.get(request.request_id, fp)
        if replay is not None:
            return replay

        order = self.orders.get(request.order_id)
        if order is None:
            raise OrderNotFound(request.order_id)
        self.ledger.open_account(order.order_id, order.paid_total)

        now = self.clock()
        for attempt in range(self.max_attempts):
            view = self.ledger.view(order.order_id)
            decision = decide(order, request, view, now, self.policy, self._signals(order, now), self.rules)
            refund: Optional[Refund] = None
            if not decision.approved:
                break
            try:
                refund, decision = self._execute(order, request, decision, now, view.version)
                break
            except ConcurrencyConflict:
                # another instance changed the account between our read and our write: re-decide
                # on the fresh view (the amount may now need capping, or nothing may be left)
                continue
        else:
            decision = Decision(Outcome.ESCALATE, None, ["concurrency_retries_exhausted"], self.policy.version)
            refund = None

        result = RefundResult(decision, refund)
        self.idempotency.put(request.request_id, fp, result)
        return result

    # ---------------------------------------------------------------- helpers
    def _execute(self, order: Order, request: RefundRequest, decision: Decision, now: datetime,
                 expected_version: int):
        assert decision.amount is not None
        try:
            refund = self.ledger.reserve(order.order_id, request.request_id, decision.amount,
                                         decision.policy_version, now, decision.reasons,
                                         expected_version=expected_version)
        except OverRefund as e:
            # Defence in depth: the policy said yes but the ledger said no. Never pay.
            return None, Decision(Outcome.REJECT, None, decision.reasons + [f"ledger_guard:{e}"], decision.policy_version)
        if refund.status is not RefundStatus.PENDING:
            return refund, decision  # retry of an already-executed reservation
        try:
            receipt = self.provider.refund(request.request_id, refund.amount)
        except ProviderDeclined as e:
            self.ledger.fail(refund.refund_id)
            return refund, Decision(Outcome.REJECT, None, decision.reasons + [f"provider_declined:{e}"], decision.policy_version)
        except ProviderTimeout:
            self.ledger.mark_unknown(refund.refund_id)
            return refund, decision
        self.ledger.settle(refund.refund_id, receipt.provider_ref)
        return refund, decision

    def _signals(self, order: Order, now: datetime) -> Signals:
        since = now - timedelta(days=self.policy.history_window_days)
        order_ids = [o.order_id for o in self.orders.for_customer(order.customer_id)]
        n = sum(
            1 for r in self.ledger.refunds_for_orders(order_ids)
            if r.status in (RefundStatus.SETTLED, RefundStatus.PENDING, RefundStatus.UNKNOWN) and r.created_at >= since
        )
        return Signals(customer_refunds_in_window=n, abuse_score=self.abuse_scores(order.customer_id))
