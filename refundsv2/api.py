"""Thin HTTP layer. Amounts arrive as decimal strings; floats are a 422 at the door."""
from __future__ import annotations

import time
from typing import Callable, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from refundsv2.db import connect
from refundsv2.ledger import UnknownOrder, get_order, insert_order, ledger_view, list_refunds
from refundsv2.model import Order, OrderStatus, Reason, Refund, RefundRequest
from refundsv2.money import InvalidAmount, format_cents, parse_amount
from refundsv2.policy import DAY, PolicyConfig
from refundsv2.provider import FakeProvider
from refundsv2.service import DuplicateRequestConflict, PolicyVersionConflict, RefundService


class RefundRequestIn(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    order_id: str
    amount: str = Field(description="decimal string, e.g. '40.00'")
    reason: Reason
    currency: str = "USD"
    abuse_score: Optional[float] = Field(default=None, ge=0, le=1)

    @field_validator("amount", mode="before")
    @classmethod
    def _no_floats(cls, v):
        if isinstance(v, float):
            raise ValueError("amount must be a decimal string, not a float")
        return str(v)


class RefundOut(BaseModel):
    request_id: str
    order_id: str
    status: str
    amount: str
    outcome: str
    reasons: List[str]
    policy_version: str
    provider_ref: Optional[str]
    attempts: int
    trace: List[dict]

    @classmethod
    def of(cls, r: Refund) -> "RefundOut":
        return cls(request_id=r.request_id, order_id=r.order_id, status=r.status.value, amount=format_cents(r.amount_cents),
                   outcome=r.decision.outcome.value, reasons=list(r.decision.reasons), policy_version=r.decision.policy_version,
                   provider_ref=r.provider_ref, attempts=r.attempts,
                   trace=[{"rule": t.name, "passed": t.passed, "reason": t.reason} for t in r.decision.trace])


def create_app(service: RefundService, clock: Callable[[], float] = time.time) -> FastAPI:
    """`clock` is injected so tests and replays run against a fixed time; production uses the wall clock."""
    app = FastAPI(title="Automatic refunds v2")

    @app.get("/health")
    def health():
        return {"status": "ok", "policy_version": service.policy.version}

    @app.post("/refund-requests", response_model=RefundOut)
    def submit(body: RefundRequestIn):
        try:
            cents = parse_amount(body.amount)
        except InvalidAmount as e:
            raise HTTPException(422, str(e))
        now = clock()
        req = RefundRequest(body.request_id, body.order_id, cents, body.currency, body.reason, now)
        try:
            return RefundOut.of(service.handle(req, now, abuse_score=body.abuse_score))
        except UnknownOrder:
            raise HTTPException(404, f"order {body.order_id} not found")
        except DuplicateRequestConflict:
            raise HTTPException(409, f"request_id {body.request_id} was already used with a different payload")

    @app.get("/orders/{order_id}/refunds")
    def refunds(order_id: str):
        if get_order(service.conn, order_id) is None:
            raise HTTPException(404, "order not found")
        v = ledger_view(service.conn, order_id)
        return {"order_id": order_id, "paid": format_cents(v.paid_cents), "held": format_cents(v.held_cents),
                "remaining": format_cents(v.remaining_cents), "refunds": [RefundOut.of(r) for r in list_refunds(service.conn, order_id)]}

    @app.put("/admin/policy", response_model=PolicyConfig)
    def put_policy(cfg: PolicyConfig):
        try:
            return service.set_policy(cfg, clock())
        except PolicyVersionConflict as e:
            raise HTTPException(409, str(e))

    @app.post("/admin/reconcile")
    def reconcile():
        return {"actions": [{"request_id": r, "action": a} for r, a in service.reconcile(clock())]}

    return app


def demo_app() -> FastAPI:
    """uvicorn refundsv2.api:app  -> in-memory database with four demo orders."""
    conn = connect()
    now = time.time()
    insert_order(conn, Order("o_1", "c_1", 12_000, "USD", OrderStatus.DELIVERED, now - 10 * DAY, now - 8 * DAY, now - 5 * DAY))
    insert_order(conn, Order("o_2", "c_1", 45_000, "USD", OrderStatus.DELIVERED, now - 10 * DAY, now - 8 * DAY, now - 5 * DAY))
    insert_order(conn, Order("o_3", "c_2", 8_000, "USD", OrderStatus.PAID, now - 1 * DAY))
    insert_order(conn, Order("o_4", "c_3", 8_000, "USD", OrderStatus.DELIVERED, now - 90 * DAY, now - 88 * DAY, now - 80 * DAY))
    provider = FakeProvider(script={"req-timeout": ["timeout_moved"]})
    return create_app(RefundService(conn, provider, now=now))


app = demo_app()
