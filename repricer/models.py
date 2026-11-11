"""Domain models shared across extraction, engine, execution and API."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ReasonCode(StrEnum):
    # strategy
    MATCH_LOWEST = "MATCH_LOWEST"
    UNDERCUT = "UNDERCUT"
    HOLD_STRATEGY = "HOLD_STRATEGY"
    TARGET_MARGIN = "TARGET_MARGIN"
    NO_COMPETITOR_DATA = "NO_COMPETITOR_DATA"
    # bounds / adjustments
    CLAMP_MAX_DECREASE = "CLAMP_MAX_DECREASE"
    CLAMP_MAX_INCREASE = "CLAMP_MAX_INCREASE"
    FLOOR_APPLIED = "FLOOR_APPLIED"
    LOW_STOCK_NO_DECREASE = "LOW_STOCK_NO_DECREASE"
    ROUNDED = "ROUNDED"
    SHIPPING_UNKNOWN = "SHIPPING_UNKNOWN"
    NO_CHANGE = "NO_CHANGE"
    # hard guardrails (blocked)
    FLOOR_ABOVE_CURRENT = "FLOOR_ABOVE_CURRENT"
    COST_ANOMALY = "COST_ANOMALY"
    SINGLE_SOURCE_LOW_CONF = "SINGLE_SOURCE_LOW_CONF"
    DELTA_EXCEEDS_HARD_CAP = "DELTA_EXCEEDS_HARD_CAP"
    PRICE_OUT_OF_RANGE = "PRICE_OUT_OF_RANGE"
    # extraction validation
    PRICE_NOT_IN_SOURCE = "PRICE_NOT_IN_SOURCE"
    COUPON_NOT_IN_SOURCE = "COUPON_NOT_IN_SOURCE"
    ZERO_PRICE = "ZERO_PRICE"
    NO_PRICE_FOUND = "NO_PRICE_FOUND"
    IMPLAUSIBLE_PRICE = "IMPLAUSIBLE_PRICE"
    SKU_UNRESOLVED = "SKU_UNRESOLVED"
    NON_USD = "NON_USD"
    AMBIGUOUS_COUPON = "AMBIGUOUS_COUPON"
    BUNDLE_PRICE = "BUNDLE_PRICE"
    LLM_SCHEMA_INVALID = "LLM_SCHEMA_INVALID"


GUARDRAIL_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.FLOOR_ABOVE_CURRENT,
        ReasonCode.COST_ANOMALY,
        ReasonCode.SINGLE_SOURCE_LOW_CONF,
        ReasonCode.DELTA_EXCEEDS_HARD_CAP,
        ReasonCode.PRICE_OUT_OF_RANGE,
    }
)


class Strategy(StrEnum):
    MATCH_LOWEST = "match_lowest"
    UNDERCUT_LOWEST = "undercut_lowest"
    HOLD = "hold"
    TARGET_MARGIN = "target_margin"


class Decision(StrEnum):
    PROPOSE = "propose"
    HOLD = "hold"
    BLOCKED = "blocked"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    EXPIRED = "expired"


class ReviewAction(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    OVERRIDE = "override"


class OfferStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"
    SUPERSEDED = "superseded"


class ChangeSource(StrEnum):
    REVIEW_APPROVE = "review_approve"
    OVERRIDE = "override"
    SEED = "seed"
    MIGRATION = "migration"


Currency = Literal["USD", "CAD", "EUR", "GBP", "OTHER", "UNKNOWN"]
Condition = Literal["new", "used", "refurbished", "unknown"]


class ExtractedOffer(BaseModel):
    """What an extractor (LLM or regex) is allowed to say about a scrape. Entities only.

    No arithmetic fields: the extractor reports prices *as written*. Effective price is
    computed in code from these fields.
    """

    model_config = ConfigDict(extra="forbid")

    scrape_id: int
    sku_hint: str | None = Field(default=None, description="SKU or identifier visible in text")
    listed_price: Decimal | None = Field(
        default=None, description="Price as written; None if no price appears", ge=0
    )
    currency: Currency = "UNKNOWN"
    coupon_amount: Decimal | None = Field(default=None, ge=0)
    coupon_percent: Decimal | None = Field(default=None, ge=0, le=100)
    shipping: Decimal | None = Field(
        default=None, description="None = not mentioned; 0 = explicitly free", ge=0
    )
    condition: Condition = "unknown"
    in_stock: bool | None = None
    is_bundle: bool = Field(default=False, description="Price covers multiple units")
    confidence: int = Field(ge=0, le=100)


class InventorySnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    sku: str
    title: str
    cost_price_cents: int
    min_margin_bp: int
    current_price_cents: int
    stock_count: int
    price_class: str


class EligibleOffer(BaseModel):
    """A competitor offer as seen by the rules engine."""

    model_config = ConfigDict(frozen=True)

    offer_id: int
    competitor: str
    effective_price_cents: int
    confidence: int
    scraped_at: datetime
    shipping_unknown: bool = False


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: Strategy
    undercut_cents: int = Field(default=0, ge=0)
    undercut_bp: int = Field(default=0, ge=0)
    target_margin_bp: int = Field(default=0, ge=0)
    max_decrease_bp: int = Field(ge=0, le=10_000)
    max_increase_bp: int = Field(ge=0)
    min_competitor_confidence: int = Field(ge=0, le=100)
    max_offer_age_hours: int = Field(gt=0)
    low_stock_threshold: int = Field(default=0, ge=0)
    round_to_cents: int = Field(default=0, ge=0, le=99)


class Recommendation(BaseModel):
    """Pure output of the rules engine. Ids/timestamps are assigned on persist."""

    model_config = ConfigDict(frozen=True)

    sku: str
    current_price_cents: int
    cost_price_cents: int
    min_margin_bp: int
    floor_price_cents: int
    lowest_competitor_cents: int | None
    competitor_offer_ids: tuple[int, ...]
    raw_target_cents: int
    proposed_price_cents: int
    delta_cents: int
    delta_bp: int
    decision: Decision
    reason_codes: tuple[ReasonCode, ...]
    explanation: str
