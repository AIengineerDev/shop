"""Domain model. Plain frozen dataclasses: no I/O, no framework types."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

from refunds.money import Money


class OrderStatus(str, Enum):
    PENDING_PAYMENT = "pending_payment"
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


REFUNDABLE_STATUSES = frozenset({OrderStatus.PAID, OrderStatus.SHIPPED, OrderStatus.DELIVERED})


class RefundReason(str, Enum):
    NOT_RECEIVED = "not_received"
    DAMAGED = "damaged"
    WRONG_ITEM = "wrong_item"
    CHANGED_MIND = "changed_mind"
    OTHER = "other"


class Outcome(str, Enum):
    APPROVE = "approve"      # refund automatically, possibly for a capped amount
    ESCALATE = "escalate"    # queue for a human; no money moves
    REJECT = "reject"        # malformed or impossible; no money moves


class RefundStatus(str, Enum):
    PENDING = "pending"      # reserved in the ledger, provider not yet confirmed
    SETTLED = "settled"      # provider confirmed the money moved
    FAILED = "failed"        # provider declined; reservation released
    UNKNOWN = "unknown"      # provider call timed out; reservation held until reconciled


@dataclass(frozen=True)
class Order:
    order_id: str
    customer_id: str
    paid_total: Money
    status: OrderStatus
    paid_at: datetime
    delivered_at: Optional[datetime] = None


@dataclass(frozen=True)
class RefundRequest:
    request_id: str            # client-supplied idempotency key
    order_id: str
    amount: Money
    reason: RefundReason
    submitted_at: datetime

    def fingerprint(self) -> str:
        """What must be identical for a retry to count as the *same* request."""
        return f"{self.order_id}|{self.amount}|{self.reason.value}"


@dataclass(frozen=True)
class LedgerView:
    """Read-only snapshot of an order's refund account, passed *into* the pure decision."""
    paid: Money
    settled: Money
    reserved: Money            # PENDING + UNKNOWN: money that may already be gone
    version: int = 0

    @property
    def remaining(self) -> Money:
        return self.paid - self.settled - self.reserved


@dataclass(frozen=True)
class Signals:
    """External inputs to the policy that are not part of the order itself."""
    customer_refunds_in_window: int = 0
    abuse_score: Optional[float] = None


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    amount: Optional[Money]           # amount to refund when APPROVE, else None
    reasons: List[str]
    policy_version: str

    @property
    def approved(self) -> bool:
        return self.outcome is Outcome.APPROVE


@dataclass
class Refund:
    refund_id: str
    request_id: str
    order_id: str
    amount: Money
    status: RefundStatus
    policy_version: str
    created_at: datetime
    provider_ref: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
