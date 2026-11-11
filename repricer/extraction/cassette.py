"""Record/replay of LLM responses keyed by a hash of the request body.

Replay is the default in tests so CI needs no API key. A missing cassette in replay mode
raises ``CassetteMissing`` (not ``ExtractorUnavailable``) so it fails loudly instead of
silently falling back to the regex extractor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class CassetteMissing(LookupError):
    pass


def request_key(body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


class CassetteStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def load(self, body: dict[str, Any]) -> str:
        key = request_key(body)
        p = self.path(key)
        if not p.exists():
            raise CassetteMissing(
                f"no cassette {key} in {self.directory}; record with REPRICER_LLM_MODE=record"
            )
        data = json.loads(p.read_text())
        return str(data["response_text"])

    def save(self, body: dict[str, Any], response_text: str, meta: dict[str, Any]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        p = self.path(request_key(body))
        p.write_text(
            json.dumps(
                {"request": body, "response_text": response_text, "meta": meta},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return p
