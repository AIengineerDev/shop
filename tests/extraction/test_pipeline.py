from __future__ import annotations

import socket
import sqlite3
from datetime import datetime

import pytest

from repricer.db import repo
from repricer.extraction.pipeline import extract_stage
from repricer.extraction.regex_extractor import RegexExtractor
from repricer.extraction.types import ExtractionResult, ExtractorUnavailable, ScrapeInput


def _count(conn: sqlite3.Connection, status: str | None = None) -> int:
    if status:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM competitor_offers WHERE status = ?", (status,)
            ).fetchone()[0]
        )
    return int(conn.execute("SELECT COUNT(*) FROM competitor_offers").fetchone()[0])


def test_ac_e3_regex_pipeline_offline(
    seeded_db: sqlite3.Connection, frozen_now: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_net(*a: object, **k: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", no_net)
    summary = extract_stage(seeded_db, extractor=RegexExtractor(), now=frozen_now)
    assert summary.scrapes == 200
    assert summary.accepted + summary.rejected + summary.needs_review == 200
    assert summary.accepted >= 150
    rows = seeded_db.execute(
        "SELECT DISTINCT extractor, confidence FROM competitor_offers"
    ).fetchall()
    assert [tuple(r) for r in rows] == [("regex:v1", 40)] or all(
        r[0] == "regex:v1" and r[1] in (40, 0) for r in rows
    )
    # worked example: scrape 1 -> 37.99 effective, accepted, new
    o = seeded_db.execute("SELECT * FROM competitor_offers WHERE scrape_id = 1").fetchone()
    assert o["status"] == "accepted" and o["effective_price_cents"] == 3799
    assert o["condition"] == "new" and o["sku"] == "SKU-0001"
    # sku_in_text scrapes resolved from the text
    unresolved = seeded_db.execute(
        "SELECT COUNT(*) FROM competitor_offers WHERE reject_reason = 'SKU_UNRESOLVED'"
    ).fetchone()[0]
    assert unresolved == 0


def test_ac_e5_idempotent_and_force(seeded_db: sqlite3.Connection, frozen_now: datetime) -> None:
    extract_stage(seeded_db, extractor=RegexExtractor(), now=frozen_now)
    n = _count(seeded_db)
    again = extract_stage(seeded_db, extractor=RegexExtractor(), now=frozen_now)
    assert again.scrapes == 0 and _count(seeded_db) == n
    forced = extract_stage(seeded_db, extractor=RegexExtractor(), now=frozen_now, force=True)
    assert forced.scrapes == 200
    assert _count(seeded_db) == 2 * n
    assert _count(seeded_db, "superseded") == n
    assert len(repo.unextracted_scrapes(seeded_db)) == 0


class Flaky:
    name = "llm:test"

    def extract(self, items: list[ScrapeInput]) -> list[ExtractionResult]:
        raise ExtractorUnavailable("503")


def test_fallback_on_unavailable(seeded_db: sqlite3.Connection, frozen_now: datetime) -> None:
    s = extract_stage(
        seeded_db, extractor=Flaky(), now=frozen_now, batch_size=50, fallback=RegexExtractor()
    )
    assert s.fallback_batches == 4
    assert _count(seeded_db) == 200
    with pytest.raises(ExtractorUnavailable):
        extract_stage(seeded_db, extractor=Flaky(), now=frozen_now, force=True)
