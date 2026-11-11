"""Extraction stage: scrapes -> extractor -> validation -> offers. Idempotent."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from repricer.clock import parse_iso
from repricer.db import repo
from repricer.db.connection import transaction
from repricer.extraction.effective_price import effective_price
from repricer.extraction.sku_resolve import resolve_sku
from repricer.extraction.types import (
    ExtractionResult,
    Extractor,
    ExtractorUnavailable,
    ScrapeInput,
)
from repricer.extraction.validate import validate
from repricer.models import ExtractedOffer, OfferStatus, ReasonCode
from repricer.money import MoneyError, to_bp, to_cents

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractSummary:
    scrapes: int
    accepted: int
    rejected: int
    needs_review: int
    fallback_batches: int


def _persist(
    conn: sqlite3.Connection,
    scrape: sqlite3.Row,
    result: ExtractionResult,
    extractor_name: str,
    titles: list[tuple[str, str]],
    now: datetime,
) -> OfferStatus:
    offer = result.offer
    sku = resolve_sku(
        scrape_sku=scrape["sku"],
        sku_hint=offer.sku_hint if offer else None,
        raw_text=scrape["raw_text"],
        titles=titles,
    )
    current: int | None = None
    if sku is not None:
        current = repo.get_inventory(conn, sku).current_price_cents

    if offer is None:
        reason = ReasonCode.LLM_SCHEMA_INVALID if "schema" in (result.error or "") else None
        reason = reason or ReasonCode.NO_PRICE_FOUND
        status = OfferStatus.REJECTED
        eff: int | None = None
        codes: list[ReasonCode] = []
        listed: int | None = None
        coupon_c = 0
        coupon_bp = 0
        ship: int | None = None
        currency = "UNKNOWN"
        condition = "unknown"
        in_stock: bool | None = None
        confidence = 0
    else:
        v = validate(offer, scrape["raw_text"], resolved_sku=sku, current_price_cents=current)
        status, reason = v.status, v.reject_reason
        codes = list(v.codes)
        eff, eff_codes = effective_price(offer)
        codes.extend(eff_codes)
        try:
            listed = None if offer.listed_price is None else to_cents(offer.listed_price)
            coupon_c = 0 if offer.coupon_amount is None else to_cents(offer.coupon_amount)
            coupon_bp = 0 if offer.coupon_percent is None else to_bp(offer.coupon_percent)
            ship = None if offer.shipping is None else to_cents(offer.shipping)
        except MoneyError as e:
            status, reason = OfferStatus.REJECTED, ReasonCode.LLM_SCHEMA_INVALID
            codes.append(ReasonCode.LLM_SCHEMA_INVALID)
            listed, coupon_c, coupon_bp, ship, eff = None, 0, 0, None, None
            log.warning("scrape %s: sub-cent amount from extractor: %s", scrape["scrape_id"], e)
        currency = offer.currency
        condition = offer.condition
        in_stock = offer.in_stock
        confidence = min(offer.confidence, 90)

    repo.insert_offer(
        conn,
        scrape_id=scrape["scrape_id"],
        sku=sku,
        competitor=scrape["competitor"],
        listed_price_cents=listed,
        coupon_cents=coupon_c,
        coupon_bp=coupon_bp,
        shipping_cents=ship,
        effective_price_cents=eff if status is not OfferStatus.REJECTED else None,
        currency=currency,
        condition=condition,
        in_stock=in_stock,
        scraped_at=parse_iso(scrape["scraped_at"]),
        extractor=extractor_name,
        confidence=confidence,
        status=status,
        reject_reason=None if reason is None else reason.value,
        reason_codes=[c.value for c in codes],
        raw_llm_json=result.raw_json,
        now=now,
    )
    repo.mark_scrape_extracted(conn, scrape["scrape_id"])
    return status


def extract_stage(
    conn: sqlite3.Connection,
    *,
    extractor: Extractor,
    now: datetime,
    batch_size: int = 20,
    fallback: Extractor | None = None,
    force: bool = False,
) -> ExtractSummary:
    scrapes = repo.all_scrapes(conn) if force else repo.unextracted_scrapes(conn)
    titles = repo.list_inventory_titles(conn)
    counts = {OfferStatus.ACCEPTED: 0, OfferStatus.REJECTED: 0, OfferStatus.NEEDS_REVIEW: 0}
    fallbacks = 0
    for i in range(0, len(scrapes), batch_size):
        batch = scrapes[i : i + batch_size]
        inputs = [ScrapeInput(s["scrape_id"], s["raw_text"]) for s in batch]
        name = extractor.name
        try:
            results = extractor.extract(inputs)
        except ExtractorUnavailable as e:
            if fallback is None:
                raise
            log.warning("extractor %s unavailable (%s); falling back to %s", name, e, fallback.name)
            results = fallback.extract(inputs)
            name = fallback.name
            fallbacks += 1
        by_id = {r.scrape_id: r for r in results}
        with transaction(conn):
            for s in batch:
                if force:
                    repo.supersede_offers_for_scrape(conn, s["scrape_id"])
                r = by_id.get(s["scrape_id"]) or ExtractionResult(
                    s["scrape_id"], None, error="extractor returned no result"
                )
                st = _persist(conn, s, r, name, titles, now)
                counts[st] = counts.get(st, 0) + 1
    return ExtractSummary(
        scrapes=len(scrapes),
        accepted=counts[OfferStatus.ACCEPTED],
        rejected=counts[OfferStatus.REJECTED],
        needs_review=counts[OfferStatus.NEEDS_REVIEW],
        fallback_batches=fallbacks,
    )


__all__ = ["ExtractSummary", "ExtractedOffer", "extract_stage"]
