"""Typed contracts shared by every pipeline stage. No LLM imports here."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypeAliasType

# An explicit recursive JSON type: unlike Any / pydantic.JsonValue it renders as a fully typed
# OpenAPI schema (AC-4.1: no Any-typed fields) and rejects non-JSON values.
JsonData = TypeAliasType("JsonData", "str | int | float | bool | None | list[JsonData] | dict[str, JsonData]")

AttrType = Literal["enum", "int", "float", "string", "bool"]
AttrValue = str | int | float | bool | list[str]
Severity = Literal["blocking", "warning", "info"]
Stage = Literal["clean", "extract", "candidates", "classify", "reconcile", "validate", "comply", "route"]
ClassificationSource = Literal["agree", "llm", "deterministic"]
Status = Literal["draft", "quarantined"]


class AttributeSpec(BaseModel):
    """One entry of canonical_categories.required_attributes_json."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    type: AttrType
    allowed: list[str] | None = None
    min: float | None = None
    max: float | None = None
    max_len: int | None = None
    multi: bool = False
    pattern: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> AttributeSpec:
        if self.type == "enum" and not self.allowed:
            raise ValueError(f"enum attribute {self.key!r} needs a non-empty 'allowed' list")
        if self.type != "enum" and self.allowed is not None:
            raise ValueError(f"'allowed' is only valid for enum attributes ({self.key!r})")
        if self.type not in ("int", "float") and (self.min is not None or self.max is not None):
            raise ValueError(f"'min'/'max' only valid for numeric attributes ({self.key!r})")
        if self.type != "string" and (self.max_len is not None or self.pattern is not None):
            raise ValueError(f"'max_len'/'pattern' only valid for string attributes ({self.key!r})")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min > max for {self.key!r}")
        return self


class ReasonCode(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    severity: Severity
    stage: Stage
    detail: str | None = None


class RawFeedRow(BaseModel):
    """A row of incoming_supplier_feeds; extra columns are kept in `extra`."""

    feed_item_id: str
    supplier_id: str
    raw_title: str | None = None
    raw_bullets: str | None = None
    raw_tags: str | None = None
    raw_price: str | None = None
    received_at: str | None = None
    extra: dict[str, JsonData] = Field(default_factory=dict)


class NormalizedListing(BaseModel):
    feed_item_id: str
    supplier_id: str
    clean_title: str
    bullets: list[str]
    tags: dict[str, str]
    unmapped_tags: dict[str, str]
    det_attrs: dict[str, AttrValue]
    keyword_tail: list[str]
    price_min: Decimal | None
    price_max: Decimal | None
    reason_codes: list[ReasonCode]
    raw_json: dict[str, JsonData]

    @property
    def search_text(self) -> str:
        """Text used for candidate retrieval. The keyword tail is marketing noise and is excluded."""
        return " ".join([self.clean_title, *self.bullets])

    @property
    def compliance_text(self) -> str:
        """Everything the supplier wrote, normalized: cleaned title, keyword tail, bullets, raw title, tags."""
        raw_title = str(self.raw_json.get("raw_title") or "")
        return " ".join([self.clean_title, *self.keyword_tail, *self.bullets, raw_title, *self.tags.values()])


class Candidate(BaseModel):
    category_id: str
    path: str
    score: float


class LLMAttempt(BaseModel):
    purpose: Literal["classify", "extract"]
    attempt: int
    raw_response: str | None
    error: str | None
    ok: bool


class LLMClassification(BaseModel):
    """Base shape; the runtime builds a subclass whose category_id is a Literal of candidates."""

    model_config = ConfigDict(extra="forbid")

    category_id: str
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(max_length=500)


class CandidateListing(BaseModel):
    normalized: NormalizedListing
    category_id: str | None
    classification_source: ClassificationSource | None
    det_top1: Candidate | None
    llm_classification: LLMClassification | None
    attributes: dict[str, AttrValue]
    extra_attributes: dict[str, AttrValue]
    llm_trace: list[LLMAttempt]
    reason_codes: list[ReasonCode]


class ValidationResult(BaseModel):
    reason_codes: list[ReasonCode]

    @property
    def blocking(self) -> list[ReasonCode]:
        return [r for r in self.reason_codes if r.severity == "blocking"]

    @property
    def ok(self) -> bool:
        return not self.blocking


class DraftListing(BaseModel):
    listing_id: int
    feed_item_id: str
    supplier_id: str
    category_id: str
    clean_title: str
    attributes: dict[str, AttrValue]
    price: Decimal | None
    pipeline_run_id: str
    rules_version: str
    created_at: datetime
    reason_codes: list[ReasonCode] = Field(default_factory=list)


class QuarantineItem(BaseModel):
    quarantine_id: int
    feed_item_id: str
    supplier_id: str
    proposed_category_id: str | None
    candidate: CandidateListing | None
    reason_codes: list[ReasonCode]
    pipeline_run_id: str
    rules_version: str
    created_at: datetime
    resolved_at: datetime | None = None
    resolution: str | None = None


class PipelineRunRecord(BaseModel):
    run_id: str
    feed_item_id: str
    stage: Stage | Literal["error", "revalidate"]
    seq: int
    input: dict[str, JsonData] | None
    output: dict[str, JsonData] | None
    llm_raw: list[dict[str, JsonData]] | None
    duration_ms: int
    created_at: datetime
