"""Environment-driven settings. No secrets are read here except the API key by the SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ExtractorMode = Literal["llm", "regex"]

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_POLICIES_PATH = PACKAGE_DIR / "rules" / "policies.yaml"
DEFAULT_MODEL = "claude-sonnet-5-5"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    db_path: Path
    model: str
    extractor: ExtractorMode
    live_llm: bool
    hard_cap_bp: int
    llm_batch: int
    policies_path: Path
    cassette_dir: Path

    @classmethod
    def from_env(cls) -> Settings:
        extractor_raw = os.environ.get("REPRICER_EXTRACTOR", "llm").strip().lower() or "llm"
        if extractor_raw not in ("llm", "regex"):
            raise ValueError(f"REPRICER_EXTRACTOR must be 'llm' or 'regex', got {extractor_raw!r}")
        extractor: ExtractorMode = "llm" if extractor_raw == "llm" else "regex"
        policies_raw = os.environ.get("REPRICER_POLICIES_PATH", "").strip()
        cassette_raw = os.environ.get("REPRICER_CASSETTE_DIR", "").strip()
        return cls(
            db_path=Path(os.environ.get("REPRICER_DB_PATH", "dev.db")),
            model=os.environ.get("REPRICER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            extractor=extractor,
            live_llm=_env_bool("REPRICER_LIVE_LLM", False),
            hard_cap_bp=_env_int("REPRICER_HARD_CAP_BP", 5000),
            llm_batch=_env_int("REPRICER_LLM_BATCH", 20),
            policies_path=Path(policies_raw) if policies_raw else DEFAULT_POLICIES_PATH,
            cassette_dir=Path(cassette_raw)
            if cassette_raw
            else Path("tests") / "fixtures" / "llm_cassettes",
        )
