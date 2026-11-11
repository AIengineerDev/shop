from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from repricer.models import ExtractedOffer


@dataclass(frozen=True)
class ScrapeInput:
    """The *only* thing an extractor is given. No cost, margin or current price."""

    scrape_id: int
    raw_text: str


@dataclass(frozen=True)
class ExtractionResult:
    scrape_id: int
    offer: ExtractedOffer | None
    raw_json: str | None = None
    error: str | None = None


class ExtractorUnavailable(RuntimeError):
    """Raised when an extractor cannot run at all (network/API). Triggers fallback."""


class Extractor(Protocol):
    name: str

    def extract(self, items: list[ScrapeInput]) -> list[ExtractionResult]: ...
