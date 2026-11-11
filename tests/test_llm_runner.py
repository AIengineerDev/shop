import json

from catalog_onboard.models import AttributeSpec, Candidate, NormalizedListing
from catalog_onboard.pipeline.llm import LLMRequest, LLMTransportError
from catalog_onboard.pipeline.llm.fake import FakeLLM, ScriptedLLM
from catalog_onboard.pipeline.llm.runner import classify, extract


def listing(**kw) -> NormalizedListing:
    base = dict(
        feed_item_id="x",
        supplier_id="s",
        clean_title="Men Gym Shorts",
        bullets=["Material: polyester"],
        tags={},
        unmapped_tags={},
        det_attrs={},
        keyword_tail=[],
        price_min=None,
        price_max=None,
        reason_codes=[],
        raw_json={},
    )
    base.update(kw)
    return NormalizedListing(**base)


CANDS = [
    Candidate(category_id="a.shorts", path="A > Shorts", score=2.0),
    Candidate(category_id="a.jeans", path="A > Jeans", score=1.0),
]


def test_ac_2_2_retry_then_give_up():
    llm = ScriptedLLM(
        [
            "not json at all",
            json.dumps({"category_id": "zzz", "confidence": 0.9, "rationale": "r"}),
            json.dumps({"category_id": "a.jeans", "confidence": 0.9, "rationale": "r"}),
        ]
    )
    parsed, attempts = classify(llm, listing(), CANDS, max_retries=1)
    assert parsed is None
    assert len(attempts) == 2 and not any(a.ok for a in attempts)
    assert "invalid JSON" in attempts[0].error and "schema violation" in attempts[1].error
    assert attempts[1].raw_response and "zzz" in attempts[1].raw_response
    # retry prompt carries the validator message back to the model
    assert "rejected by the schema validator" in llm.requests[1].user


def test_retry_succeeds_on_second_attempt():
    llm = ScriptedLLM(["{", json.dumps({"category_id": "a.jeans", "confidence": 0.7, "rationale": "r"})])
    parsed, attempts = classify(llm, listing(), CANDS, max_retries=2)
    assert parsed is not None and parsed.category_id == "a.jeans"
    assert [a.ok for a in attempts] == [False, True]


def test_transport_error_counts_as_attempt():
    class Flaky:
        name = "flaky"

        def complete(self, request: LLMRequest) -> str:
            raise LLMTransportError("boom")

    parsed, attempts = classify(Flaky(), listing(), CANDS, max_retries=1)
    assert parsed is None and len(attempts) == 2 and all("transport" in a.error for a in attempts)


def test_fake_llm_is_deterministic_and_schema_valid():
    specs = [
        AttributeSpec(key="color", type="enum", allowed=["black", "navy"], multi=True),
        AttributeSpec(key="gender", type="enum", allowed=["men", "women"]),
        AttributeSpec(key="material", type="string", max_len=64),
    ]
    lst = listing(clean_title="Men Black Navy Shorts")
    a, _ = extract(FakeLLM(), lst, specs, max_retries=0)
    b, _ = extract(FakeLLM(), lst, specs, max_retries=0)
    assert a is not None and a.model_dump() == b.model_dump()
    assert a.model_dump() == {"color": ["black", "navy"], "gender": "men", "material": "polyester"}
    c, _ = classify(FakeLLM(), lst, CANDS, max_retries=0)
    assert c is not None and c.category_id == "a.shorts"


def test_anthropic_client_request_shape(monkeypatch):
    """No network: assert the request we would send, using a stub SDK client."""
    from types import SimpleNamespace

    from catalog_onboard.pipeline.llm.anthropic_client import FALLBACK_BETA, AnthropicClient

    captured = {}

    class _Msgs:
        def create(self, **kw):
            captured.update(kw)
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text='{"ok":1}')])

    stub = SimpleNamespace(beta=SimpleNamespace(messages=_Msgs()))
    client = AnthropicClient(model="claude-opus-5-5", max_tokens=99, client=stub)
    req = LLMRequest(purpose="classify", system="S", user="U", json_schema={"type": "object"}, hints={"secret": 1})
    assert client.complete(req) == '{"ok":1}'
    assert captured["model"] == "claude-opus-5-5" and captured["max_tokens"] == 99
    assert captured["system"] == "S" and captured["messages"] == [{"role": "user", "content": "U"}]
    assert captured["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert captured["fallbacks"] == "default" and captured["betas"] == [FALLBACK_BETA]
    assert "hints" not in captured and "secret" not in json.dumps(captured)


def test_anthropic_client_refusal_is_transport_error():
    from types import SimpleNamespace

    import pytest

    from catalog_onboard.pipeline.llm.anthropic_client import AnthropicClient

    class _Msgs:
        def create(self, **kw):
            return SimpleNamespace(stop_reason="refusal", stop_details={"category": "x"}, content=[])

    client = AnthropicClient(model="m", client=SimpleNamespace(beta=SimpleNamespace(messages=_Msgs())))
    with pytest.raises(LLMTransportError, match="refused"):
        client.complete(LLMRequest(purpose="classify", system="", user="", json_schema={}))
