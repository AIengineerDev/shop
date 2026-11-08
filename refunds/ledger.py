"""The refund ledger: the ONE place the over-refund invariant is enforced.

    settled + reserved + new_amount <= paid        (reserved = PENDING + UNKNOWN)

Every refund goes through `reserve()` before any money moves; the provider is only called with a
reservation in hand. Reservations are keyed by request_id, so a retry whose idempotency record was
lost still cannot reserve twice. The in-memory implementation stands in for a database table.

Concurrency (several instances, one shared database): each order account carries a `version`.
`reserve(expected_version=...)` is a compare-and-set: if another instance reserved against the order
since the caller took its LedgerView, the write is refused with ConcurrencyConflict and the caller
re-reads and re-decides. The lock below plays the role of the database's row-level atomicity; the SQL
equivalent is

    UPDATE refund_accounts SET reserved = reserved + :amt, version = version + 1
     WHERE order_id = :id AND version = :expected AND settled + reserved + :amt <= paid;
    -- 0 rows updated -> conflict or over-refund; re-read and decide again
    INSERT INTO refunds (request_id UNIQUE, ...)   -- the unique key is the second line of defence

An alternative is SELECT ... FOR UPDATE on the account row; CAS is preferred because it needs no
lock held across the decision and degrades to a retry rather than a wait.
"""
from __future__ import annotations

import threading

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


class ConcurrencyConflict(LedgerError):
    """The account changed since the caller's LedgerView was taken."""


HOLDS = frozenset({RefundStatus.PENDING, RefundStatus.UNKNOWN})


@dataclass
class _Account:
    paid: Money
    refunds: List[Refund]
    version: int = 0


class RefundLedger:
    def __init__(self) -> None:
        self._accounts: Dict[str, _Account] = {}
        self._by_request: Dict[str, Refund] = {}
        self._by_id: Dict[str, Refund] = {}
        self._lock = threading.Lock()

    # -------------------------------------------------------------- setup
    def open_account(self, order_id: str, paid: Money) -> None:
        with self._lock:
            self._accounts.setdefault(order_id, _Account(paid, []))

    # -------------------------------------------------------------- reads
    def view(self, order_id: str) -> LedgerView:
        with self._lock:
            return self._view_locked(self._account(order_id))

    def _view_locked(self, acc: _Account) -> LedgerView:
        settled = Money.zero(acc.paid.currency)
        reserved = Money.zero(acc.paid.currency)
        for r in acc.refunds:
            if r.status is RefundStatus.SETTLED:
                settled = settled + r.amount
            elif r.status in HOLDS:
                reserved = reserved + r.amount
        return LedgerView(acc.paid, settled, reserved, acc.version)

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
                reasons: Optional[List[str]] = None, expected_version: Optional[int] = None) -> Refund:
        """Reserve `amount` against the order. Idempotent on request_id.

        Raises OverRefund if the invariant would break, ConcurrencyConflict if `expected_version`
        is given and the account has moved on since. The whole check-and-write is atomic.
        """
        with self._lock:
            existing = self._by_request.get(request_id)
            if existing is not None:
                return existing
            acc = self._account(order_id)
            if not amount.is_positive:
                raise LedgerError(f"reserve amount must be positive, got {amount}")
            if expected_version is not None and acc.version != expected_version:
                raise ConcurrencyConflict(f"order {order_id}: version {acc.version} != expected {expected_version}")
            view = self._view_locked(acc)
            if view.settled + view.reserved + amount > acc.paid:
                raise OverRefund(f"order {order_id}: {view.settled}+{view.reserved}+{amount} > paid {acc.paid}")
            return self._append_locked(acc, order_id, request_id, amount, policy_version, now, reasons)

    def _append_locked(self, acc: _Account, order_id: str, request_id: str, amount: Money,
                       policy_version: str, now: datetime, reasons: Optional[List[str]]) -> Refund:
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
        acc.version += 1
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
        with self._lock:
            r = self._by_id[refund_id]
            if r.status is RefundStatus.SETTLED and status is not RefundStatus.SETTLED:
                raise LedgerError(f"{refund_id} already settled; cannot move to {status.value}")
            if r.status is RefundStatus.FAILED:
                raise LedgerError(f"{refund_id} already failed; cannot move to {status.value}")
            r.status = status
            if provider_ref is not None:
                r.provider_ref = provider_ref
            self._accounts[r.order_id].version += 1  # a release changes what is refundable
            return r
