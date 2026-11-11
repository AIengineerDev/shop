"""Record LLM cassettes for the golden fixtures. Needs ANTHROPIC_API_KEY.

    REPRICER_LLM_MODE=record .venv/bin/python scripts/record_cassettes.py

Batches are built exactly like the pipeline does (same order, same batch size), so the
replayed keys match what tests/extraction/test_llm.py requests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from repricer.config import Settings
from repricer.extraction.llm import ClaudeExtractor
from repricer.extraction.types import ScrapeInput

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "scrapes"


def fixture_inputs() -> list[ScrapeInput]:
    items: list[ScrapeInput] = []
    for f in sorted(FIXTURES.glob("*.json")):
        for row in json.loads(f.read_text()):
            items.append(ScrapeInput(row["scrape_id"], row["raw_text"]))
    items.sort(key=lambda i: i.scrape_id)
    return items


def main() -> int:
    s = Settings.from_env()
    if s.llm_mode != "record":
        print("set REPRICER_LLM_MODE=record", file=sys.stderr)
        return 2
    ex = ClaudeExtractor(model=s.model, mode="record", cassette_dir=s.cassette_dir)
    items = fixture_inputs()
    n = 0
    for i in range(0, len(items), s.llm_batch):
        batch = items[i : i + s.llm_batch]
        res = ex.extract(batch)
        n += sum(1 for r in res if r.offer is not None)
    print(f"recorded {len(items)} scrapes -> {n} parsed offers into {s.cassette_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
