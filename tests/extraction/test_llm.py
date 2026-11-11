from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest

from repricer.extraction.cassette import CassetteMissing, CassetteStore, request_key
from repricer.extraction.llm import (
    MAX_CONFIDENCE,
    SYSTEM_PROMPT,
    ClaudeExtractor,
    build_request,
    parse_response_text,
)
from repricer.extraction.types import ExtractorUnavailable, ScrapeInput
from repricer.seed import generate

CASSETTES = Path(__file__).parent.parent / "fixtures" / "llm_cassettes"
ITEMS = [
    ScrapeInput(1, "Brand new Acme Wireless Headphones X1 on sale for $42.99 with $5 coupon"),
    ScrapeInput(2, "Compact Blender C3 - currently unavailable"),
]


class FakeMessages:
    def __init__(
        self, text: str | None = None, exc: Exception | None = None, stop: str = "end_turn"
    ):
        self.text, self.exc, self.stop = text, exc, stop
        self.calls: list[dict[str, Any]] = []

    def create(self, **body: Any) -> Any:
        self.calls.append(body)
        if self.exc:
            raise self.exc
        return SimpleNamespace(
            stop_reason=self.stop,
            content=[SimpleNamespace(type="text", text=self.text or "")],
        )


def fake_client(messages: FakeMessages) -> Any:
    return SimpleNamespace(messages=messages)


GOOD = json.dumps(
    {
        "offers": [
            {
                "scrape_id": 1,
                "sku_hint": None,
                "listed_price": "42.99",
                "currency": "USD",
                "coupon_amount": "5.00",
                "coupon_percent": None,
                "shipping": None,
                "condition": "new",
                "in_stock": None,
                "is_bundle": False,
                "confidence": 97,
            },
            {
                "scrape_id": 2,
                "sku_hint": None,
                "listed_price": None,
                "currency": "UNKNOWN",
                "coupon_amount": None,
                "coupon_percent": None,
                "shipping": None,
                "condition": "unknown",
                "in_stock": False,
                "is_bundle": False,
                "confidence": 80,
            },
        ]
    }
)


def test_ac_e4_request_never_contains_inventory_data() -> None:
    """The LLM sees scrape text only. No cost, margin, or current price anywhere."""
    data = generate(seed=42)
    items = [ScrapeInput(s.scrape_id, s.raw_text) for s in data.scrapes[:40]]
    body = build_request("claude-sonnet-5-5", items)
    serialized = json.dumps(body).lower()
    for forbidden in ("cost", "margin", "current_price", "stock_count", "min_margin"):
        assert forbidden not in serialized, forbidden
    for inv in data.inventory[:50]:
        assert f"{inv.cost_price}" not in serialized or f"${inv.cost_price}" in serialized
    assert "arithmetic" in SYSTEM_PROMPT.lower()
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "tool_choice" not in body and "temperature" not in body


def test_parse_caps_confidence_and_handles_nulls() -> None:
    res = parse_response_text(GOOD, ITEMS)
    assert res[0].offer is not None and res[0].offer.confidence == MAX_CONFIDENCE
    assert res[0].offer.coupon_amount is not None and str(res[0].offer.coupon_amount) == "5.00"
    assert res[1].offer is not None and res[1].offer.listed_price is None
    assert res[1].offer.in_stock is False


def test_parse_schema_failures_do_not_raise() -> None:
    res = parse_response_text("not json", ITEMS)
    assert all(r.offer is None and r.error and "schema" in r.error for r in res)
    bad = json.dumps({"offers": [{"scrape_id": 1, "listed_price": "abc", "confidence": 5}]})
    res = parse_response_text(bad, ITEMS)
    assert res[0].offer is None and "schema" in (res[0].error or "")
    assert res[1].offer is None and "missing" in (res[1].error or "")


def test_live_mode_uses_client_and_maps_errors() -> None:
    fm = FakeMessages(text=GOOD)
    ex = ClaudeExtractor(model="m", mode="live", client=fake_client(fm))
    res = ex.extract(ITEMS)
    assert res[0].offer is not None
    assert fm.calls[0]["model"] == "m" and fm.calls[0]["system"][0]["cache_control"]

    import httpx2 as httpx

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.APIStatusError("boom", response=httpx.Response(503, request=req), body=None)
    ex = ClaudeExtractor(model="m", mode="live", client=fake_client(FakeMessages(exc=err)))
    with pytest.raises(ExtractorUnavailable):
        ex.extract(ITEMS)

    ex = ClaudeExtractor(
        model="m", mode="live", client=fake_client(FakeMessages(text=GOOD, stop="refusal"))
    )
    with pytest.raises(ExtractorUnavailable):
        ex.extract(ITEMS)


def test_record_then_replay(tmp_path: Path) -> None:
    fm = FakeMessages(text=GOOD)
    rec = ClaudeExtractor(model="m", mode="record", cassette_dir=tmp_path, client=fake_client(fm))
    rec.extract(ITEMS)
    assert len(list(tmp_path.glob("*.json"))) == 1
    rep = ClaudeExtractor(model="m", mode="replay", cassette_dir=tmp_path)
    res = rep.extract(ITEMS)
    assert res[0].offer is not None and len(fm.calls) == 1
    with pytest.raises(CassetteMissing):
        rep.extract([ScrapeInput(99, "other")])
    assert (
        request_key(build_request("m", ITEMS))
        == CassetteStore(tmp_path).path(request_key(build_request("m", ITEMS))).stem
    )


def test_ac_e1_golden_llm_replay() -> None:
    """Replays recorded cassettes against the golden fixtures. Skips until recorded."""
    if not any(CASSETTES.glob("*.json")):
        pytest.skip("no LLM cassettes recorded; run scripts/record_cassettes.py with an API key")
    from repricer.config import Settings
    from scripts.record_cassettes import fixture_inputs

    s = Settings.from_env()
    ex = ClaudeExtractor(model=s.model, mode="replay", cassette_dir=CASSETTES)
    items = fixture_inputs()
    fixtures = {}
    for f in sorted((CASSETTES.parent / "scrapes").glob("*.json")):
        for row in json.loads(f.read_text()):
            fixtures[row["scrape_id"]] = row["expected"]
    mismatches = []
    for i in range(0, len(items), s.llm_batch):
        for r in ex.extract(items[i : i + s.llm_batch]):
            exp = fixtures[r.scrape_id]
            if r.offer is None:
                if exp["listed_price"] is not None:
                    mismatches.append((r.scrape_id, "no offer"))
                continue
            got = r.offer.model_dump()
            for field in ("listed_price", "coupon_amount", "coupon_percent", "shipping"):
                g, e = got[field], exp[field]
                if (g is None) != (e is None) or (g is not None and float(g) != float(e)):
                    mismatches.append((r.scrape_id, field, str(g), e))
            if got["currency"] != exp["currency"] or got["is_bundle"] != exp["is_bundle"]:
                mismatches.append((r.scrape_id, "currency/bundle"))
    # allow a small tolerance for genuinely ambiguous snippets
    assert len(mismatches) <= len(items) * 0.03, mismatches[:20]
