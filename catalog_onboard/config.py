"""Runtime settings, read from environment variables and an optional `.env` file (spec §8).

`.env` (path: $ENV_FILE, default ./.env) is loaded into os.environ without overriding variables
that are already set, so the Anthropic SDK sees an API key defined there too.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

LLMProvider = Literal["anthropic", "fake"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

DEFAULT_RULES_DIR = Path(__file__).parent / "rules"


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    return float(raw) if raw else default


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    return int(raw) if raw else default


def parse_dotenv(text: str) -> dict[str, str]:
    """Minimal .env parser: KEY=VALUE lines, optional `export `, # comments, single/double quotes."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            out[key] = value
    return out


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Load `.env` into os.environ without overriding existing variables. Returns what it read."""
    env_path = path or Path(os.environ.get("ENV_FILE", ".env"))
    if not env_path.is_file():
        return {}
    values = parse_dotenv(env_path.read_text(encoding="utf-8"))
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values


@dataclass(frozen=True)
class Settings:
    llm_provider: LLMProvider = "fake"
    # Spec §8: latest Sonnet-class model. Classification / extraction is short structured output,
    # so effort defaults to "low"; thinking still counts against max_tokens, hence 4096.
    llm_model: str = "claude-sonnet-5-5"
    llm_effort: Effort = "low"
    llm_max_retries: int = 2
    llm_max_tokens: int = 4096
    high_conf_threshold: float = 0.75
    disagree_quarantine_below: float = 0.8
    candidate_k: int = 8
    rules_dir: Path = field(default=DEFAULT_RULES_DIR)

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        load_dotenv(env_file)
        env = os.environ
        provider_raw = env.get("LLM_PROVIDER")
        if provider_raw is None:
            provider: LLMProvider = "anthropic" if env.get("ANTHROPIC_API_KEY") else "fake"
        elif provider_raw in ("anthropic", "fake"):
            provider = provider_raw  # type: ignore[assignment]
        else:
            raise ValueError(f"LLM_PROVIDER must be 'anthropic' or 'fake', got {provider_raw!r}")
        effort_raw = env.get("LLM_EFFORT", cls.llm_effort)
        if effort_raw not in EFFORTS:
            raise ValueError(f"LLM_EFFORT must be one of {EFFORTS}, got {effort_raw!r}")
        return cls(
            llm_provider=provider,
            llm_model=env.get("LLM_MODEL", cls.llm_model),
            llm_effort=effort_raw,  # type: ignore[arg-type]
            llm_max_retries=_env_int(env, "LLM_MAX_RETRIES", cls.llm_max_retries),
            llm_max_tokens=_env_int(env, "LLM_MAX_TOKENS", cls.llm_max_tokens),
            high_conf_threshold=_env_float(env, "HIGH_CONF_THRESHOLD", cls.high_conf_threshold),
            disagree_quarantine_below=_env_float(env, "DISAGREE_QUARANTINE_BELOW", cls.disagree_quarantine_below),
            candidate_k=_env_int(env, "CANDIDATE_K", cls.candidate_k),
            rules_dir=Path(env.get("RULES_DIR", str(DEFAULT_RULES_DIR))),
        )
