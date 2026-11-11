"""Stage 5b: deterministic compliance checks. Never asks a model anything."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable
from decimal import Decimal

from catalog_onboard.models import AttrValue, ReasonCode

MIN_TITLE_LEN = 3


def _blocking(code: str, detail: str | None = None) -> ReasonCode:
    return ReasonCode(code=code, severity="blocking", stage="comply", detail=detail)


def keyword_pattern(keyword: str) -> re.Pattern[str]:
    parts = [re.escape(p) for p in keyword.lower().split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)", re.IGNORECASE)


def find_prohibited(text: str, keywords: Iterable[str]) -> list[str]:
    hits: list[str] = []
    for kw in keywords:
        if kw and keyword_pattern(kw).search(text):
            hits.append(kw)
    return hits


def fingerprint(supplier_id: str, clean_title: str, attrs: dict[str, AttrValue]) -> str:
    payload = json.dumps([supplier_id, clean_title.lower(), attrs], sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def check_compliance(
    *,
    clean_title: str,
    scan_text: str,
    category_prohibited: Iterable[str],
    global_prohibited: Iterable[str],
    price_min: Decimal | None,
    duplicate_exists: Callable[[], bool],
) -> list[ReasonCode]:
    """Run every check (no short-circuit) so the full list of codes is captured.

    `scan_text` must be the *full* normalized supplier text (raw title incl. keyword tail + bullets),
    not the cleaned title: cleaning strips marketing tails, which is exactly where sellers hide words.
    """
    codes: list[ReasonCode] = []
    for kw in find_prohibited(scan_text, category_prohibited):
        codes.append(_blocking(f"PROHIBITED_KEYWORD:{kw}"))
    for kw in find_prohibited(scan_text, global_prohibited):
        codes.append(_blocking(f"PROHIBITED_GLOBAL:{kw}"))
    if price_min is None or price_min <= 0:
        codes.append(_blocking("PRICE_INVALID", None if price_min is None else str(price_min)))
    if len(clean_title.strip()) < MIN_TITLE_LEN:
        codes.append(_blocking("TITLE_TOO_SHORT", clean_title))
    if duplicate_exists():
        codes.append(_blocking("DUPLICATE_LISTING"))
    return codes
