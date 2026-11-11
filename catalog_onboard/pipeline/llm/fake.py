"""Deterministic stand-ins for the real model. Used in CI and when no API key is configured."""

from __future__ import annotations

import json
import re
from collections import deque
from typing import Any

from catalog_onboard.pipeline.llm import LLMRequest


class FakeLLM:
    """Heuristic responder: picks the top candidate; extracts enum values by substring match."""

    name = "fake"

    def complete(self, request: LLMRequest) -> str:
        if request.purpose == "classify":
            cands: list[str] = list(request.hints.get("candidate_ids", []))
            if not cands:
                return "{}"
            return json.dumps({"category_id": cands[0], "confidence": 0.9, "rationale": "best lexical match (fake)"})
        text = str(request.hints.get("text", "")).lower()
        specs: list[dict[str, Any]] = list(request.hints.get("specs", []))
        out: dict[str, Any] = {}
        for spec in specs:
            key, typ = spec["key"], spec["type"]
            val: Any = None
            if typ == "enum":
                hits = [a for a in spec.get("allowed", []) if re.search(rf"(?<!\w){re.escape(a.lower())}(?!\w)", text)]
                if hits:
                    val = hits if spec.get("multi") else hits[0]
            elif typ == "string" and key == "material":
                m = re.search(r"(?:material|made of|fabric)\s*:?\s*([a-z ]+)", text)
                if m:
                    val = m.group(1).strip()[: spec.get("max_len") or 64]
            out[key] = val
        return json.dumps(out)


class ScriptedLLM:
    """Returns pre-recorded responses in order; raises when exhausted. For tests."""

    name = "scripted"

    def __init__(self, responses: list[str]):
        self.queue: deque[str] = deque(responses)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        self.requests.append(request)
        if not self.queue:
            raise AssertionError("ScriptedLLM exhausted")
        return self.queue.popleft()


class ForbiddenLLM:
    """Fails loudly if invoked. Used to prove a stage makes no LLM calls."""

    name = "forbidden"

    def complete(self, request: LLMRequest) -> str:
        raise AssertionError(f"LLM must not be called here (purpose={request.purpose})")
