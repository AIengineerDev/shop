"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

LLMProvider = Literal["anthropic", "fake"]

DEFAULT_RULES_DIR = Path(__file__).parent / "rules"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    llm_provider: LLMProvider = "fake"
    llm_model: str = "claude-opus-5-5"
    llm_max_retries: int = 2
    llm_max_tokens: int = 1024
    high_conf_threshold: float = 0.75
    disagree_quarantine_below: float = 0.8
    candidate_k: int = 8
    rules_dir: Path = field(default=DEFAULT_RULES_DIR)

    @classmethod
    def from_env(cls) -> Settings:
        provider_raw = os.environ.get("LLM_PROVIDER")
        if provider_raw is None:
            provider: LLMProvider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "fake"
        elif provider_raw in ("anthropic", "fake"):
            provider = provider_raw  # type: ignore[assignment]
        else:
            raise ValueError(f"LLM_PROVIDER must be 'anthropic' or 'fake', got {provider_raw!r}")
        return cls(
            llm_provider=provider,
            llm_model=os.environ.get("LLM_MODEL", cls.llm_model),
            llm_max_retries=_env_int("LLM_MAX_RETRIES", cls.llm_max_retries),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", cls.llm_max_tokens),
            high_conf_threshold=_env_float("HIGH_CONF_THRESHOLD", cls.high_conf_threshold),
            disagree_quarantine_below=_env_float("DISAGREE_QUARANTINE_BELOW", cls.disagree_quarantine_below),
            candidate_k=_env_int("CANDIDATE_K", cls.candidate_k),
            rules_dir=Path(os.environ.get("RULES_DIR", str(DEFAULT_RULES_DIR))),
        )
