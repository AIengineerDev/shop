"""Response/request models. Every money field is a 2-decimal string, never a number."""

from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from repricer.money import fmt

Money = str  # "42.99"


def money(cents: int | None) -> Money | None:
    return None if cents is None else fmt(cents)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stages: list[str] = Field(default_factory=lambda: ["extract", "recommend"])


class RunOut(BaseModel):
    run_id: str
    status: str
    trigger: str
    started_at: str
    finished_at: str | None
    policy_version: int
    skus_considered: int
    recommendations: int
    offers_extracted: int
    error: str | None

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> RunOut:
        return cls(**{k: r[k] for k in cls.model_fields})


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actor: str = Field(min_length=1, max_length=200)
    note: str | None = Field(default=None, max_length=2000)


class OverrideRequest(ReviewRequest):
    price: Decimal = Field(description='Decimal string, e.g. "41.99"', gt=0)


class OfferOut(BaseModel):
    offer_id: int
    competitor: str
    raw_text: str
    listed_price: Money | None
    coupon: Money
    coupon_percent: str
    shipping: Money | None
    effective_price: Money | None
    currency: str
    condition: str
    in_stock: bool | None
    scraped_at: str
    extractor: str
    confidence: int
    status: str
    reject_reason: str | None
    reason_codes: list[str]

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> OfferOut:
        return cls(
            offer_id=r["offer_id"],
            competitor=r["competitor"],
            raw_text=r["raw_text"],
            listed_price=money(r["listed_price_cents"]),
            coupon=fmt(r["coupon_cents"]),
            coupon_percent=f"{Decimal(r['coupon_bp']) / 100:.2f}",
            shipping=money(r["shipping_cents"]),
            effective_price=money(r["effective_price_cents"]),
            currency=r["currency"],
            condition=r["condition"],
            in_stock=None if r["in_stock"] is None else bool(r["in_stock"]),
            scraped_at=r["scraped_at"],
            extractor=r["extractor"],
            confidence=r["confidence"],
            status=r["status"],
            reject_reason=r["reject_reason"],
            reason_codes=json.loads(r["reason_codes"] or "[]"),
        )


class RecommendationOut(BaseModel):
    recommendation_id: str
    run_id: str
    sku: str
    decision: str
    review_status: str
    current_price: Money
    proposed_price: Money
    floor_price: Money
    cost_price: Money
    min_margin_percent: str
    lowest_competitor_price: Money | None
    raw_target_price: Money
    delta: Money
    delta_bp: int
    reason_codes: list[str]
    explanation: str
    policy_version: int
    created_at: str

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> RecommendationOut:
        return cls(
            recommendation_id=r["recommendation_id"],
            run_id=r["run_id"],
            sku=r["sku"],
            decision=r["decision"],
            review_status=r["review_status"],
            current_price=fmt(r["current_price_cents"]),
            proposed_price=fmt(r["proposed_price_cents"]),
            floor_price=fmt(r["floor_price_cents"]),
            cost_price=fmt(r["cost_price_cents"]),
            min_margin_percent=f"{Decimal(r['min_margin_bp']) / 100:.2f}",
            lowest_competitor_price=money(r["lowest_competitor_cents"]),
            raw_target_price=fmt(r["raw_target_cents"]),
            delta=fmt(r["delta_cents"]),
            delta_bp=r["delta_bp"],
            reason_codes=json.loads(r["reason_codes"]),
            explanation=r["explanation"],
            policy_version=r["policy_version"],
            created_at=r["created_at"],
        )


class ActionOut(BaseModel):
    action_id: str
    action: str
    actor: str
    override_price: Money | None
    note: str | None
    created_at: str


class ReviewCard(RecommendationOut):
    """Everything a human needs to decide without opening the DB."""

    title: str
    stock_count: int
    price_class: str
    eligible_offers: list[OfferOut]
    actions: list[ActionOut]


class Page(BaseModel):
    items: list[RecommendationOut]
    total: int
    limit: int
    offset: int


class ReviewResult(BaseModel):
    recommendation_id: str
    review_status: str
    change_id: str | None
    sku: str
    new_price: Money | None


class ChangeOut(BaseModel):
    change_id: str
    sku: str
    recommendation_id: str | None
    action_id: str | None
    old_price: Money
    new_price: Money
    source: str
    actor: str
    applied_at: str
    prev_change_id: str | None
    row_hash: str
    decision: str | None = None
    reason_codes: list[str] | None = None
    action: str | None = None
    note: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChangeOut:
        return cls(
            change_id=d["change_id"],
            sku=d["sku"],
            recommendation_id=d["recommendation_id"],
            action_id=d["action_id"],
            old_price=fmt(d["old_price_cents"]),
            new_price=fmt(d["new_price_cents"]),
            source=d["source"],
            actor=d["actor"],
            applied_at=d["applied_at"],
            prev_change_id=d["prev_change_id"],
            row_hash=d["row_hash"],
            decision=d.get("decision"),
            reason_codes=d.get("reason_codes"),
            action=d.get("action"),
            note=d.get("note"),
        )


class BreakOut(BaseModel):
    sku: str
    change_id: str
    reason: str


class VerifyOut(BaseModel):
    ok: bool
    breaks: list[BreakOut]


class PoliciesOut(BaseModel):
    version: int
    policies: dict[str, dict[str, Any]]
