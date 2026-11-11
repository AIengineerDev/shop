"""Prompt builders. These must never mention compliance, prohibited keywords, or policy."""

from __future__ import annotations

import json
from collections.abc import Sequence

from catalog_onboard.models import AttributeSpec, Candidate, NormalizedListing

CLASSIFY_SYSTEM = (
    "You classify e-commerce product listings into a marketplace taxonomy. "
    "You will be given a cleaned listing and a short list of candidate leaf categories. "
    "Choose exactly one category_id from the candidates. Give a confidence between 0 and 1 "
    "and a one-sentence rationale. Respond with JSON only."
)

EXTRACT_SYSTEM = (
    "You extract product attributes from e-commerce listings. "
    "Only report an attribute when the listing states it or it is unambiguous from the product type. "
    "Use null when the listing does not say. Respond with JSON only."
)


def listing_block(listing: NormalizedListing) -> str:
    lines = [f"Title: {listing.clean_title}"]
    if listing.bullets:
        lines.append("Bullets:")
        lines.extend(f"- {b}" for b in listing.bullets)
    if listing.keyword_tail:
        lines.append("Keywords: " + " ".join(listing.keyword_tail))
    if listing.det_attrs:
        lines.append("Known attributes: " + json.dumps(listing.det_attrs, sort_keys=True, default=str))
    return "\n".join(lines)


def classify_user(listing: NormalizedListing, candidates: Sequence[Candidate]) -> str:
    cand_lines = "\n".join(f"- {c.category_id}  ({c.path})" for c in candidates)
    return f"{listing_block(listing)}\n\nCandidate categories (choose one category_id):\n{cand_lines}"


def extract_user(listing: NormalizedListing, specs: Sequence[AttributeSpec]) -> str:
    spec_lines = []
    for s in specs:
        desc: str = s.type
        if s.allowed:
            desc += " one of " + ", ".join(s.allowed)
            if s.multi:
                desc += " (list, may contain several)"
        if s.min is not None or s.max is not None:
            desc += f" in [{s.min}, {s.max}]"
        if s.max_len is not None:
            desc += f" (max {s.max_len} chars)"
        spec_lines.append(f"- {s.key}: {desc}")
    return f"{listing_block(listing)}\n\nAttributes to extract (null if not stated):\n" + "\n".join(spec_lines)


def retry_suffix(error: str) -> str:
    return f"\n\nYour previous answer was rejected by the schema validator:\n{error}\nReturn corrected JSON only."
