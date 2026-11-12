"""Stage 1: raw feed row -> NormalizedListing. Pure function of (row, rules)."""

from __future__ import annotations

import unicodedata
from typing import Any

from catalog_onboard.models import AttrValue, NormalizedListing, RawFeedRow, ReasonCode, Severity
from catalog_onboard.pipeline import extract as ex
from catalog_onboard.rules import Rules

MAX_TITLE_LEN = 150


def _code(code: str, severity: Severity, detail: str | None = None) -> ReasonCode:
    return ReasonCode(code=code, severity=severity, stage="clean", detail=detail)


def _merge_attr(
    attrs: dict[str, AttrValue], key: str, value: AttrValue, *, source: str, codes: list[ReasonCode]
) -> None:
    """Deterministic merge: lists union; scalars keep the first value and flag conflicts."""
    if key not in attrs:
        attrs[key] = value
        return
    existing = attrs[key]
    if isinstance(existing, list) and isinstance(value, list):
        for v in value:
            if v not in existing:
                existing.append(v)
        return
    if existing != value:
        codes.append(_code(f"ATTR_CONFLICT:{key}", "warning", f"kept {existing!r}, {source} said {value!r}"))


_TEXT_FIELDS = ("raw_title", "raw_bullets", "raw_tags", "raw_price")


def _drop_undecodable(row: RawFeedRow, codes: list[ReasonCode]) -> RawFeedRow:
    """AC-1.5: a field with undecodable bytes (U+FFFD after the store's lenient decode) is unset
    and flagged; the rest of the row is still processed."""
    bad = [f for f in _TEXT_FIELDS if isinstance(getattr(row, f), str) and "\ufffd" in getattr(row, f)]
    for f in bad:
        codes.append(_code(f"TEXT_ENCODING_INVALID:{f}", "warning", "field unset: invalid UTF-8"))
    return row.model_copy(update={f: None for f in bad}) if bad else row


def is_unsupported_language(text: str) -> bool:
    """§10: English only. Letters outside the Latin script (accented Latin is fine) -> warn."""
    return any(ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN") for ch in text)


def normalize(row: RawFeedRow, rules: Rules) -> NormalizedListing:
    codes: list[ReasonCode] = []
    attrs: dict[str, AttrValue] = {}
    abbrev = ex.build_abbreviation_regex(rules)
    row = _drop_undecodable(row, codes)

    # --- title ---------------------------------------------------------------
    raw_title = row.raw_title or ""
    title = ex.normalize_text(raw_title)
    if not title:
        codes.append(_code("TITLE_EMPTY", "blocking"))
    elif is_unsupported_language(title):
        codes.append(_code("LANG_UNSUPPORTED", "warning", "non-Latin script in title"))
    title, tail = ex.extract_tail(title)
    title, colors, sizes = ex.extract_variations(title)
    title, pack = ex.extract_pack_count(title)
    title = ex.expand_abbreviations(title, rules, abbrev)
    title = ex.normalize_text(title).strip(" -,:;")
    clean_title = ex.smart_title_case(title)[:MAX_TITLE_LEN].rstrip()

    # --- bullets -------------------------------------------------------------
    bullets = [ex.expand_abbreviations(b, rules, abbrev) for b in ex.split_bullets(row.raw_bullets)]

    # --- tags (supplier-specific -> canonical) --------------------------------
    tags: dict[str, str] = {}
    unmapped: dict[str, str] = {}
    for key, value in ex.split_tag_pairs(row.raw_tags):
        canonical = rules.tag_key(row.supplier_id, key)
        if canonical is None:
            unmapped[key] = value
            codes.append(_code("TAG_UNMAPPED", "warning", f"{key}={value}"))
            continue
        tags[canonical] = value
        coerced = ex.coerce_tag_value(canonical, value, row.supplier_id, rules)
        if coerced is not None:
            _merge_attr(attrs, canonical, coerced, source="tag", codes=codes)

    # --- title-derived attributes (tags take precedence on conflicts) ---------
    if colors:
        _merge_attr(attrs, "color", list(colors), source="title", codes=codes)
    if sizes:
        _merge_attr(attrs, "size", list(sizes), source="title", codes=codes)
    if pack is not None:
        _merge_attr(attrs, "pack_count", pack, source="title", codes=codes)
    gender = ex.extract_gender(title)
    if gender:
        _merge_attr(attrs, "gender", gender, source="title", codes=codes)
    for key, val in ex.extract_capacity(" ".join([title, *bullets])).items():
        _merge_attr(attrs, key, val, source="text", codes=codes)
    material = ex.extract_material(bullets)
    if material:
        _merge_attr(attrs, "material", material, source="bullets", codes=codes)
    connector = ex.extract_connector(bullets)
    if connector:
        _merge_attr(attrs, "connector", connector, source="bullets", codes=codes)

    # --- price ---------------------------------------------------------------
    price_min, price_max, ok = ex.parse_price(row.raw_price)
    if not ok:
        codes.append(_code("PRICE_UNPARSEABLE", "warning", row.raw_price))

    raw_json: dict[str, Any] = {
        "raw_title": row.raw_title,
        "raw_bullets": row.raw_bullets,
        "raw_tags": row.raw_tags,
        "raw_price": row.raw_price,
        "received_at": row.received_at,
        **row.extra,
    }
    return NormalizedListing(
        feed_item_id=row.feed_item_id,
        supplier_id=row.supplier_id,
        clean_title=clean_title,
        bullets=bullets,
        tags=tags,
        unmapped_tags=unmapped,
        det_attrs=attrs,
        keyword_tail=tail,
        price_min=price_min,
        price_max=price_max,
        reason_codes=codes,
        raw_json=raw_json,
    )
