"""Golden fixtures against the regex extractor. Every template case must parse."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from repricer.extraction.regex_extractor import CONFIDENCE, NAME, RegexExtractor, parse
from repricer.extraction.types import ScrapeInput

FIXTURES = sorted((Path(__file__).parent.parent / "fixtures" / "scrapes").glob("*.json"))
ENTITY_FIELDS = (
    "sku_hint",
    "listed_price",
    "currency",
    "coupon_amount",
    "coupon_percent",
    "shipping",
    "condition",
    "in_stock",
    "is_bundle",
)


def _norm(v: Any) -> Any:
    if isinstance(v, Decimal):
        return str(v.quantize(Decimal("0.01")))
    if isinstance(v, str):
        try:
            return str(Decimal(v).quantize(Decimal("0.01")))
        except Exception:  # noqa: BLE001
            return v
    return v


def _cases() -> list[tuple[str, dict[str, Any]]]:
    out = []
    for f in FIXTURES:
        for item in json.loads(f.read_text()):
            out.append((f"{f.stem}#{item['scrape_id']}", item))
    return out


@pytest.mark.parametrize("name,item", _cases(), ids=[c[0] for c in _cases()])
def test_ac_e1_golden_regex(name: str, item: dict[str, Any]) -> None:
    parsed = parse(item["raw_text"])
    expected = item["expected"]
    if expected["listed_price"] is None and parsed is None:
        return
    assert parsed is not None, item["raw_text"]
    got = parsed.model_dump()
    for field in ENTITY_FIELDS:
        exp = expected[field]
        g = got[field]
        if field == "sku_hint" and exp is None:
            continue  # regex may pick up an incidental hint; only required when expected
        assert _norm(g) == _norm(exp), f"{field}: {item['raw_text']!r} -> {g!r} != {exp!r}"


def test_fixture_count_and_cases() -> None:
    cases = _cases()
    assert len(cases) >= 40
    assert len({c[0].split("#")[0] for c in cases}) >= 15


def test_extractor_shape() -> None:
    ex = RegexExtractor()
    res = ex.extract([ScrapeInput(7, "Thing $9.99"), ScrapeInput(8, "nothing here")])
    assert ex.name == NAME
    assert res[0].offer is not None and res[0].offer.scrape_id == 7
    assert res[0].offer.confidence == CONFIDENCE
    assert res[1].offer is None and res[1].error
