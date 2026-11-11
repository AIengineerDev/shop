"""Claude extractor. Entities only; no arithmetic; never sees inventory data.

Uses structured outputs (``output_config.format``) so the response is schema-valid JSON.
Amounts are returned as decimal *strings* to keep floats out of the pipeline. The model's
self-reported confidence is capped at 90 in code.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import anthropic
from pydantic import ValidationError

from repricer.config import LlmMode
from repricer.extraction.cassette import CassetteStore
from repricer.extraction.types import ExtractionResult, ExtractorUnavailable, ScrapeInput
from repricer.models import ExtractedOffer

log = logging.getLogger(__name__)

MAX_CONFIDENCE = 90
MAX_TOKENS = 8000

# NOTE: wording deliberately avoids the words that AC-E4 forbids in the request payload.
SYSTEM_PROMPT = """You extract pricing entities from raw competitor listing snippets for an \
e-commerce catalog. For each snippet, report exactly what the text says. Rules:

- Report the listed price exactly as written. If the text shows an old price and a new \
price (e.g. "was $50, now $42"), the listed price is the NEW price.
- Do NOT subtract coupons, do NOT add shipping, do NOT compute totals, do NOT convert \
currencies, do NOT do any arithmetic. Other code does that.
- coupon_amount is an absolute discount as written ("$5 coupon" -> "5.00"). coupon_percent \
is a percentage discount as written ("10% off" -> "10"). Report both only if both appear.
- shipping: null if not mentioned, "0.00" if explicitly free, else the amount as written.
- currency: USD for "$", "US$" or "USD"; CAD for "CA$"; EUR for "€"; GBP for "£"; \
UNKNOWN if there is no price; OTHER for anything else.
- condition: "new" only when the text says new/brand new; "used"; "refurbished"; otherwise \
"unknown".
- in_stock: true/false only when the text says so, else null.
- is_bundle: true when the price covers multiple units ("2 for $50").
- sku_hint: an identifier in the text such as "[SKU-0042]" or "SKU 0042", else null.
- confidence: 0-100, your certainty that listed_price is the per-unit sticker price.
- If no price appears, set listed_price to null and currency to UNKNOWN.
- Every amount is a decimal string with at most two decimal places, e.g. "42.99", "5.00".
- Return one entry per snippet, same scrape_id, same order."""

_AMOUNT = {"type": ["string", "null"], "pattern": r"^\d+(\.\d{1,2})?$"}
OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "offers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scrape_id": {"type": "integer"},
                    "sku_hint": {"type": ["string", "null"]},
                    "listed_price": _AMOUNT,
                    "currency": {
                        "type": "string",
                        "enum": ["USD", "CAD", "EUR", "GBP", "OTHER", "UNKNOWN"],
                    },
                    "coupon_amount": _AMOUNT,
                    "coupon_percent": _AMOUNT,
                    "shipping": _AMOUNT,
                    "condition": {
                        "type": "string",
                        "enum": ["new", "used", "refurbished", "unknown"],
                    },
                    "in_stock": {"type": ["boolean", "null"]},
                    "is_bundle": {"type": "boolean"},
                    "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "required": [
                    "scrape_id",
                    "sku_hint",
                    "listed_price",
                    "currency",
                    "coupon_amount",
                    "coupon_percent",
                    "shipping",
                    "condition",
                    "in_stock",
                    "is_bundle",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["offers"],
    "additionalProperties": False,
}


def build_request(model: str, items: list[ScrapeInput]) -> dict[str, Any]:
    """The full request body. Built once so cassettes and the no-leak test share it."""
    payload = [{"scrape_id": i.scrape_id, "raw_text": i.raw_text} for i in items]
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": "Snippets as JSON:\n" + json.dumps(payload, ensure_ascii=False),
            }
        ],
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
    }


def _dec(v: Any) -> Decimal | None:
    if v is None:
        return None
    return Decimal(str(v))


def parse_response_text(text: str, items: list[ScrapeInput]) -> list[ExtractionResult]:
    """Turn the model's JSON into ExtractionResults. Schema failures become errors, not raises."""
    wanted = {i.scrape_id for i in items}
    try:
        data = json.loads(text)
        entries = data["offers"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return [
            ExtractionResult(i.scrape_id, None, raw_json=text, error=f"schema: {e}") for i in items
        ]
    out: dict[int, ExtractionResult] = {}
    for entry in entries:
        raw = json.dumps(entry, sort_keys=True)
        try:
            sid = int(entry["scrape_id"])
            offer = ExtractedOffer(
                scrape_id=sid,
                sku_hint=entry.get("sku_hint"),
                listed_price=_dec(entry.get("listed_price")),
                currency=entry.get("currency", "UNKNOWN"),
                coupon_amount=_dec(entry.get("coupon_amount")),
                coupon_percent=_dec(entry.get("coupon_percent")),
                shipping=_dec(entry.get("shipping")),
                condition=entry.get("condition", "unknown"),
                in_stock=entry.get("in_stock"),
                is_bundle=bool(entry.get("is_bundle", False)),
                confidence=min(int(entry.get("confidence", 0)), MAX_CONFIDENCE),
            )
        except (ValidationError, InvalidOperation, KeyError, TypeError, ValueError) as e:
            sid = int(entry.get("scrape_id", -1)) if isinstance(entry, dict) else -1
            out[sid] = ExtractionResult(sid, None, raw_json=raw, error=f"schema: {e}")
            continue
        if sid not in wanted:
            log.warning("model returned unknown scrape_id %s; ignored", sid)
            continue
        out[sid] = ExtractionResult(sid, offer, raw_json=raw)
    return [
        out.get(i.scrape_id)
        or ExtractionResult(i.scrape_id, None, raw_json=None, error="schema: missing entry")
        for i in items
    ]


class ClaudeExtractor:
    def __init__(
        self,
        *,
        model: str,
        mode: LlmMode = "live",
        cassette_dir: Path | None = None,
        client: anthropic.Anthropic | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.mode = mode
        self.name = f"llm:{model}"
        self.store = CassetteStore(cassette_dir) if cassette_dir is not None else None
        if mode != "live" and self.store is None:
            raise ValueError("record/replay modes need a cassette_dir")
        self._client = client
        self._max_retries = max_retries

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(max_retries=self._max_retries)
        return self._client

    def _call(self, body: dict[str, Any]) -> str:
        try:
            response = self.client.messages.create(**body)
        except anthropic.RateLimitError as e:
            raise ExtractorUnavailable(f"rate limited after retries: {e}") from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise ExtractorUnavailable(f"server error {e.status_code}") from e
            raise
        except anthropic.APIConnectionError as e:
            raise ExtractorUnavailable(f"connection error: {e}") from e
        if response.stop_reason == "refusal":
            raise ExtractorUnavailable("model refused the extraction request")
        if response.stop_reason == "max_tokens":
            raise ExtractorUnavailable("response truncated at max_tokens; reduce batch size")
        text = next((b.text for b in response.content if b.type == "text"), "")
        if not text:
            raise ExtractorUnavailable("empty response")
        return text

    def extract(self, items: list[ScrapeInput]) -> list[ExtractionResult]:
        if not items:
            return []
        body = build_request(self.model, items)
        if self.mode == "replay":
            assert self.store is not None
            text = self.store.load(body)
        else:
            text = self._call(body)
            if self.mode == "record":
                assert self.store is not None
                self.store.save(body, text, {"model": self.model})
        return parse_response_text(text, items)
