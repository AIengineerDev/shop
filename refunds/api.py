"""Thin HTTP layer. All domain logic lives in refunds.service / refunds.policy."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from refunds.config import PolicyConfig
from refunds.idempotency import IdempotencyConflict
from refunds.ledger import RefundLedger
from refunds.model import Order, OrderStatus, Refund, RefundReason, RefundRequest
from refunds.money import Money
from refunds.provider import FakeProvider
from refunds.service import OrderNotFound, OrderRepo, RefundResult, RefundService


class RefundRequestIn(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    order_id: str
    amount: str = Field(description="decimal string, e.g. '19.99' — never a float")
    currency: str = "USD"
    reason: RefundReason
    submitted_at: Optional[datetime] = None

    @field_validator("amount")
    @classmethod
    def _decimal(cls, v: str) -> str:
        try:
            Decimal(v)
        except InvalidOperation:
            raise ValueError("amount must be a decimal string")
        return v


def _refund_out(r: Refund) -> Dict[str, Any]:
    return {
        "refund_id": r.refund_id, "request_id": r.request_id, "order_id": r.order_id,
        "amount": str(r.amount.amount), "currency": r.amount.currency, "status": r.status.value,
        "provider_ref": r.provider_ref, "policy_version": r.policy_version,
        "created_at": r.created_at.isoformat(),
    }


def _result_out(res: RefundResult) -> Dict[str, Any]:
    d = res.decision
    return {
        "decision": {
            "outcome": d.outcome.value,
            "amount": str(d.amount.amount) if d.amount else None,
            "currency": d.amount.currency if d.amount else None,
            "reasons": d.reasons,
            "policy_version": d.policy_version,
        },
        "refund": _refund_out(res.refund) if res.refund else None,
    }


def create_app(service: RefundService) -> FastAPI:
    app = FastAPI(title="refunds")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "policy_version": service.policy.version}

    @app.post("/refund-requests")
    def submit(body: RefundRequestIn) -> Dict[str, Any]:
        req = RefundRequest(
            request_id=body.request_id,
            order_id=body.order_id,
            amount=Money.of(body.amount, body.currency),
            reason=body.reason,
            submitted_at=body.submitted_at or service.clock(),
        )
        try:
            return _result_out(service.handle(req))
        except IdempotencyConflict as e:
            raise HTTPException(409, str(e))
        except OrderNotFound as e:
            raise HTTPException(404, f"order {e} not found")

    @app.get("/orders/{order_id}/refunds")
    def refunds(order_id: str) -> Dict[str, Any]:
        if service.orders.get(order_id) is None:
            raise HTTPException(404, f"order {order_id} not found")
        v = service.ledger.view(order_id)
        return {
            "order_id": order_id,
            "paid": str(v.paid.amount), "settled": str(v.settled.amount),
            "reserved": str(v.reserved.amount), "remaining": str(v.remaining.amount),
            "refunds": [_refund_out(r) for r in service.ledger.refunds_for(order_id)],
        }

    return app


def demo_service() -> RefundService:
    """A service with a few orders for `uvicorn refunds.api:app`."""
    now = datetime.now(timezone.utc)
    orders = OrderRepo([
        Order("o_1", "c_1", Money.of("120.00"), OrderStatus.DELIVERED, now, now),
        Order("o_2", "c_1", Money.of("450.00"), OrderStatus.SHIPPED, now),
        Order("o_3", "c_2", Money.of("30.00"), OrderStatus.PAID, now),
        Order("o_4", "c_3", Money.of("99.99"), OrderStatus.CANCELLED, now),
    ])
    return RefundService(orders, RefundLedger(), FakeProvider("ok"), PolicyConfig())


app = create_app(demo_service())
