"""LLM boundary. Everything in this package *proposes*; nothing here decides policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Purpose = Literal["classify", "extract"]


@dataclass(frozen=True)
class LLMRequest:
    purpose: Purpose
    system: str
    user: str
    json_schema: dict[str, Any]
    hints: dict[str, Any] = field(default_factory=dict)
    """Structured copy of what the prompt says, for deterministic fakes. Never sent to a real model."""


class LLMTransportError(RuntimeError):
    """Network/API failure. The runner treats it as a failed attempt."""


class LLMClient(Protocol):
    name: str

    def complete(self, request: LLMRequest) -> str:
        """Return the raw text of the model's response (expected to be JSON)."""
        ...
