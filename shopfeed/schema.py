"""Pydantic models. Bulk data moves as pandas DataFrames with these column names."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from shopfeed.config import CATEGORIES, COUNTRIES, EVENT_TO_HEAD

EVENT_TYPES = sorted(set(EVENT_TO_HEAD) | {"follow"})
CARD_TYPES = ["product", "merchant", "drop", "order_status"]


class Merchant(BaseModel):
    merchant_id: str
    status: str = "active"  # active | suspended
    fulfilment_days: float = Field(ge=0)
    return_rate: float = Field(ge=0, le=1)
    review_score: float = Field(ge=0, le=5)

    @field_validator("status")
    @classmethod
    def _status(cls, v: str) -> str:
        if v not in ("active", "suspended"):
            raise ValueError("status must be active or suspended")
        return v


class Product(BaseModel):
    product_id: str
    merchant_id: str
    category: str
    price: float = Field(gt=0)
    currency: str = "USD"
    in_stock: bool = True
    ships_to: List[str]
    published_at: float
    is_consumable: bool = False
    repurchase_cycle_s: Optional[float] = None
    is_evergreen: bool = False

    @field_validator("category")
    @classmethod
    def _cat(cls, v: str) -> str:
        if v not in CATEGORIES:
            raise ValueError(f"unknown category {v}")
        return v


class User(BaseModel):
    user_id: str
    country: str
    blocked_merchants: List[str] = Field(default_factory=list)
    shop_pay_history: bool = False

    @field_validator("country")
    @classmethod
    def _country(cls, v: str) -> str:
        if v not in COUNTRIES:
            raise ValueError(f"unknown country {v}")
        return v


class Event(BaseModel):
    event_id: str
    user_id: str
    product_id: Optional[str] = None
    merchant_id: Optional[str] = None
    type: str
    ts: float
    impression_id: Optional[str] = None
    value: Optional[float] = None  # dwell seconds, order value, etc.

    @field_validator("type")
    @classmethod
    def _type(cls, v: str) -> str:
        if v not in EVENT_TYPES:
            raise ValueError(f"unknown event type {v}; allowed {EVENT_TYPES}")
        return v


class Impression(BaseModel):
    impression_id: str
    request_id: str
    user_id: str
    product_id: str
    card_type: str = "product"
    ts: float
    position: int = Field(ge=0)
    propensity: float = Field(gt=0, le=1)
    model_version: str
    source: str
    scores: Dict[str, float] = Field(default_factory=dict)
    value: float = 0.0
    features: List[float] = Field(default_factory=list)


class FeedRequest(BaseModel):
    user_id: str
    country: str
    session_id: str
    page: int = Field(default=0, ge=0)
    now: Optional[float] = None  # explicit clock for tests/replay; defaults to wall clock

    @field_validator("country")
    @classmethod
    def _country(cls, v: str) -> str:
        if v not in COUNTRIES:
            raise ValueError(f"unknown country {v}")
        return v


class FeedCard(BaseModel):
    impression_id: str
    product_id: str
    merchant_id: str
    card_type: str
    category: str
    price: float
    position: int
    source: str
    value: float
    is_exploration: bool
    reason: str


class FeedResponse(BaseModel):
    request_id: str
    user_id: str
    model_version: str
    fallback: bool
    cards: List[FeedCard]
    timings_ms: Dict[str, float]


class EventsIngest(BaseModel):
    events: List[Event]


class EventsIngestResponse(BaseModel):
    accepted: int
    duplicates: int
    ignored_future: int
