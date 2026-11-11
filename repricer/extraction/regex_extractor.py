"""Fallback extractor: deterministic regex, fixed low confidence (spec §4.5)."""

from __future__ import annotations

import re
from decimal import Decimal

from repricer.extraction.types import ExtractionResult, ScrapeInput
from repricer.models import Condition, Currency, ExtractedOffer

NAME = "regex:v1"
CONFIDENCE = 40

AMOUNT = r"(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
MONEY_RE = re.compile(
    r"(?P<sym>CA\$|US\$|USD\s*|\$|€|£)\s*(?P<amt>" + AMOUNT + r")"
    r"|(?P<amt2>" + AMOUNT + r")\s*(?P<sym2>USD|CAD|EUR|GBP)\b",
    re.IGNORECASE,
)
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*off", re.IGNORECASE)
FREE_SHIP_RE = re.compile(r"free shipping|ships free|shipping included", re.IGNORECASE)
BUNDLE_RE = re.compile(r"\b(\d+)\s*for\s*(?:CA\$|US\$|\$|€|£)", re.IGNORECASE)
SKU_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9\-_]{2,})\]|\b(SKU-?\d{3,})\b")
SYMBOL_CURRENCY: dict[str, Currency] = {
    "ca$": "CAD",
    "us$": "USD",
    "usd": "USD",
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "cad": "CAD",
    "eur": "EUR",
    "gbp": "GBP",
}


def _dec(s: str) -> Decimal:
    return Decimal(s.replace(",", "")).quantize(Decimal("0.01"))


def _classify(text: str, m: re.Match[str]) -> str:
    before = text[max(0, m.start() - 16) : m.start()].lower()
    after = text[m.end() : m.end() + 16].lower()
    if re.search(r"\bwas\b\s*$", before):
        return "old"
    if re.match(r"\s*(coupon|off\b|discount)", after) or re.search(
        r"coupon\s*(of|for|:)?\s*$", before
    ):
        return "coupon"
    if before.rstrip().endswith("+") or re.match(r"\s*(shipping|s&h|delivery)", after):
        return "shipping"
    return "price"


def parse(text: str) -> ExtractedOffer | None:
    """Return None when no price-like entity can be found at all."""
    listed: Decimal | None = None
    currency: Currency = "UNKNOWN"
    coupon_amount: Decimal | None = None
    shipping: Decimal | None = None
    for m in MONEY_RE.finditer(text):
        sym = (m.group("sym") or m.group("sym2") or "").strip().lower()
        amt = _dec(m.group("amt") or m.group("amt2"))
        kind = _classify(text, m)
        if kind == "old":
            continue
        if kind == "coupon":
            if coupon_amount is None:
                coupon_amount = amt
            continue
        if kind == "shipping":
            if shipping is None:
                shipping = amt
            continue
        if listed is None:
            listed = amt
            currency = SYMBOL_CURRENCY.get(sym, "UNKNOWN")
    pct = PCT_RE.search(text)
    coupon_percent = Decimal(pct.group(1)) if pct else None
    if FREE_SHIP_RE.search(text):
        shipping = Decimal("0.00")
    low = text.lower()
    condition: Condition = "unknown"
    if "refurbished" in low:
        condition = "refurbished"
    elif re.search(r"\(used\b|\bused\s*[-–]|\bpre-owned\b", low):
        condition = "used"
    elif re.search(r"\bbrand new\b|\bnew\b", low):
        condition = "new"
    in_stock: bool | None = None
    if re.search(r"\bin stock\b", low):
        in_stock = True
    elif re.search(r"unavailable|out of stock|sold out", low):
        in_stock = False
    sku_m = SKU_RE.search(text)
    sku_hint = (sku_m.group(1) or sku_m.group(2)) if sku_m else None
    if listed is None and coupon_amount is None and shipping is None and in_stock is None:
        return None
    return ExtractedOffer(
        scrape_id=0,
        sku_hint=sku_hint,
        listed_price=listed,
        currency=currency,
        coupon_amount=coupon_amount,
        coupon_percent=coupon_percent,
        shipping=shipping,
        condition=condition,
        in_stock=in_stock,
        is_bundle=bool(BUNDLE_RE.search(text)),
        confidence=CONFIDENCE,
    )


class RegexExtractor:
    name = NAME

    def extract(self, items: list[ScrapeInput]) -> list[ExtractionResult]:
        out: list[ExtractionResult] = []
        for item in items:
            parsed = parse(item.raw_text)
            if parsed is None:
                out.append(ExtractionResult(item.scrape_id, None, error="no entities found"))
                continue
            out.append(
                ExtractionResult(
                    item.scrape_id, parsed.model_copy(update={"scrape_id": item.scrape_id})
                )
            )
        return out
