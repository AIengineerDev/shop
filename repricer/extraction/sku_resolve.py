"""Resolve a scrape to an inventory SKU. Prefers refusing over guessing."""

from __future__ import annotations

import re
from collections.abc import Iterable

TOKEN_RE = re.compile(r"[a-z0-9]+")
TITLE_MATCH_THRESHOLD = 0.9


def _tokens(s: str) -> set[str]:
    return set(TOKEN_RE.findall(s.lower()))


def resolve_sku(
    *,
    scrape_sku: str | None,
    sku_hint: str | None,
    raw_text: str,
    titles: Iterable[tuple[str, str]],
) -> str | None:
    titles = list(titles)
    known = {sku.lower(): sku for sku, _ in titles}
    if scrape_sku and scrape_sku.lower() in known:
        return known[scrape_sku.lower()]
    if sku_hint and sku_hint.strip().lower() in known:
        return known[sku_hint.strip().lower()]
    # A known SKU literally present in the text.
    text_tokens = _tokens(raw_text)
    lowered = raw_text.lower()
    present = [sku for low, sku in known.items() if low in lowered]
    if len(present) == 1:
        return present[0]
    # Title token-set overlap: fraction of the title's tokens that appear in the text.
    best: list[str] = []
    best_score = 0.0
    for sku, title in titles:
        tt = _tokens(title)
        if not tt:
            continue
        score = len(tt & text_tokens) / len(tt)
        if score > best_score:
            best_score, best = score, [sku]
        elif score == best_score and score > 0:
            best.append(sku)
    if best_score >= TITLE_MATCH_THRESHOLD and len(best) == 1:
        return best[0]
    return None
