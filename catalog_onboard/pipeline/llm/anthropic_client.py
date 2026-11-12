"""Real client using the Anthropic SDK with structured (JSON schema) output."""

from __future__ import annotations

from typing import Any

import anthropic

from catalog_onboard.pipeline.llm import LLMRequest, LLMTransportError

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicClient:
    name = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int = 4096,
        effort: str = "low",
        client: anthropic.Anthropic | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.client = client or anthropic.Anthropic()

    def build_params(self, request: LLMRequest) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
            # Short structured output: low effort keeps thinking (which counts against max_tokens) small.
            "output_config": {
                "format": {"type": "json_schema", "schema": request.json_schema},
                "effort": self.effort,
            },
            "fallbacks": "default",
            "betas": [FALLBACK_BETA],
        }

    def complete(self, request: LLMRequest) -> str:
        try:
            response = self.client.beta.messages.create(**self.build_params(request))
        except anthropic.RateLimitError as exc:
            raise LLMTransportError(f"rate limited: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMTransportError(f"api error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMTransportError(f"connection error: {exc}") from exc
        if response.stop_reason == "max_tokens":
            raise LLMTransportError(f"output truncated at max_tokens={self.max_tokens}")
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMTransportError(f"model refused: {details}")
        text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), None)
        if text is None:
            raise LLMTransportError("response had no text block")
        return str(text)
