"""Domain model. Frozen dataclasses and enums, no framework types.

Every amount is integer cents. Every timestamp is a float unix time passed in explicitly, so the
decision is a pure function of its inputs and tests never touch the wall clock.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class OrderStatus(str, Enum):
    CREATED = "created"
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class Reason(str, Enum):
    DAMAGED = "damaged"
    NOT_AS_DESCRIBED = "not_as_described"
    LATE = "late"
    NOT_RECEIVED = "not_received"
    CHANGED_MIND = "changed_mind"
    OTHER = "other"


class Outcome(str, Enum):
    APPROVE = "approve"
    ESCALATE = "escalate"   # queued for a person; never rejected outright when a rule merely fails
    REJECT = "reject"       # impossible request: wrong state, bad amount, nothing left


class RefundStatus(str, Enum):
    PENDING = "pending"     # reserved, provider call not yet confirmed  -> counts toward the ceiling
    SETTLED = "settled"     # provider confirmed                          -> counts
    UNKNOWN = "unknown"     # provider timed out; money may have moved    -> counts until reconciled
    FAILED = "failed"       # provider declined                           -> released
    ESCALATED = "escalated" # decision recorded, no money reserved
    REJECTED = "rejected"   # decision recorded, no money reserved


HOLDS = (RefundStatus.PENDING, RefundStatus.SETTLED, RefundStatus.UNKNOWN)


@dataclass(frozen=True)
class Order:
    order_id: str
    customer_id: str
    paid_cents: int            # items + shipping + tax: the refund ceiling
    currency: str
    status: OrderStatus
    paid_at: Optional[float] = None
    shipped_at: Optional[float] = None
    delivered_at: Optional[float] = None


@dataclass(frozen=True)
class RefundRequest:
    request_id: str            # client-supplied idempotency key
    order_id: str
    amount_cents: int
    currency: str
    reason: Reason
    submitted_at: float

    def fingerprint(self) -> str:
        """What a retry must match. Timestamp excluded: a retry is submitted later by definition."""
        raw = f"{self.order_id}|{self.amount_cents}|{self.currency}|{self.reason.value}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class RuleResult:
    name: str
    passed: bool
    reason: str = ""                  # machine-readable, e.g. "outside_window:30d"
    effect: Optional[str] = None      # None | "reject" | "escalate" | "cap"
    cap_cents: Optional[int] = None   # only for effect == "cap"


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    amount_cents: int
    reasons: Tuple[str, ...]
    trace: Tuple[RuleResult, ...]
    policy_version: str

    def to_json(self) -> str:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Decision":
        d = json.loads(s)
        return cls(outcome=Outcome(d["outcome"]), amount_cents=d["amount_cents"], reasons=tuple(d["reasons"]),
                   trace=tuple(RuleResult(**r) for r in d["trace"]), policy_version=d["policy_version"])


@dataclass(frozen=True)
class Refund:
    request_id: str
    order_id: str
    amount_cents: int
    status: RefundStatus
    decision: Decision
    fingerprint: str
    created_at: float
    updated_at: float
    provider_ref: Optional[str] = None
    attempts: int = 0


@dataclass(frozen=True)
class LedgerView:
    """The order's money position as the decision sees it. Derived, never stored."""
    paid_cents: int
    held_cents: int       # sum of PENDING + SETTLED + UNKNOWN

    @property
    def remaining_cents(self) -> int:
        return self.paid_cents - self.held_cents


@dataclass(frozen=True)
class Signals:
    """External inputs to the decision that are not part of the order or the ledger."""
    refunds_last_90d: int = 0
    abuse_score: Optional[float] = None
