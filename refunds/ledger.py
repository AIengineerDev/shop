"""The refund ledger: the ONE place the over-refund invariant is enforced.

    settled + reserved + new_amount <= paid        (reserved = PENDING + UNKNOWN)

Every refund goes through `reserve()` before any money moves; the provider is only called with a
reservation in hand. Reservations are keyed by request_id, so a retry whose idempotency record was
lost still cannot reserve twice. The in-memory implementation stands in for a database table.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from refunds.model import LedgerView, Refund, RefundStatus
from refunds.money import Money


class LedgerError(Exception):
    pass


class UnknownOrder(LedgerError):
    pass


class OverRefund(LedgerError):
    pass


HOLDS = frozenset({RefundStatus.PENDING, RefundStatus.UNKNOWN})


@dataclass
class _Account:
    paid: Money
    refunds: List[Refund]


class RefundLedger:
    def __init__(self) -> None:
        self._accounts: Dict[str, _Account] = {}
        self._by_request: Dict[str, Refund] = {}
        self._by_id: Dict[str, Refund] = {}

    # -------------------------------------------------------------- setup
    def open_account(self, order_id: str, paid: Money) -> None:
        self._accounts.setdefault(order_id, _Account(paid, []))

    # -------------------------------------------------------------- reads
    def view(self, order_id: str) -> LedgerView:
        acc = self._account(order_id)
        settled = Money.zero(acc.paid.currency)
        reserved = Money.zero(acc.paid.currency)
        for r in acc.refunds:
            if r.status is RefundStatus.SETTLED:
                settled = settled + r.amount
            elif r.status in HOLDS:
                reserved = reserved + r.amount
        return LedgerView(acc.paid, settled, reserved)

    def refunds_for(self, order_id: str) -> List[Refund]:
        return list(self._account(order_id).refunds)

    def refunds_for_orders(self, order_ids: Iterable[str]) -> List[Refund]:
        out: List[Refund] = []
        for oid in order_ids:
            if oid in self._accounts:
                out.extend(self._accounts[oid].refunds)
        return out

    def by_request(self, request_id: str) -> Optional[Refund]:
        return self._by_request.get(request_id)

    def with_status(self, status: RefundStatus) -> List[Refund]:
        return [r for r in self._by_id.values() if r.status is status]

    # -------------------------------------------------------------- writes
    def reserve(self, order_id: str, request_id: str, amount: Money, policy_version: str, now: datetime,
                reasons: Optional[List[str]] = None) -> Refund:
        """Reserve `amount` against the order. Idempotent on request_id. Raises OverRefund."""
        existing = self._by_request.get(request_id)
        if existing is not None:
            return existing
        acc = self._account(order_id)
        if not amount.is_positive:
            raise LedgerError(f"reserve amount must be positive, got {amount}")
        view = self.view(order_id)
        if view.settled + view.reserved + amount > acc.paid:
            raise OverRefund(f"order {order_id}: {view.settled}+{view.reserved}+{amount} > paid {acc.paid}")
        refund = Refund(
            refund_id=f"rf_{uuid.uuid4().hex[:12]}",
            request_id=request_id,
            order_id=order_id,
            amount=amount,
            status=RefundStatus.PENDING,
            policy_version=policy_version,
            created_at=now,
            reasons=list(reasons or []),
        )
        acc.refunds.append(refund)
        self._by_request[request_id] = refund
        self._by_id[refund.refund_id] = refund
        return refund

    def settle(self, refund_id: str, provider_ref: str) -> Refund:
        return self._transition(refund_id, RefundStatus.SETTLED, provider_ref)

    def fail(self, refund_id: str) -> Refund:
        """Provider declined: the reservation is released (FAILED rows do not count against paid)."""
        return self._transition(refund_id, RefundStatus.FAILED, None)

    def mark_unknown(self, refund_id: str) -> Refund:
        """Provider timed out: keep the hold until reconciliation says what happened."""
        return self._transition(refund_id, RefundStatus.UNKNOWN, None)

    # -------------------------------------------------------------- internals
    def _account(self, order_id: str) -> _Account:
        try:
            return self._accounts[order_id]
        except KeyError:
            raise UnknownOrder(order_id) from None

    def _transition(self, refund_id: str, status: RefundStatus, provider_ref: Optional[str]) -> Refund:
        r = self._by_id[refund_id]
        if r.status is RefundStatus.SETTLED and status is not RefundStatus.SETTLED:
            raise LedgerError(f"{refund_id} already settled; cannot move to {status.value}")
        if r.status is RefundStatus.FAILED:
            raise LedgerError(f"{refund_id} already failed; cannot move to {status.value}")
        r.status = status
        if provider_ref is not None:
            r.provider_ref = provider_ref
        return r
