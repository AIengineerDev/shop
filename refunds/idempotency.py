"""Replay store keyed by the client's request_id.

Same key + same fingerprint -> return the stored result unchanged.
Same key + different fingerprint -> IdempotencyConflict (the client reused a key for a new request).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


class IdempotencyConflict(Exception):
    pass


@dataclass(frozen=True)
class Stored:
    fingerprint: str
    result: Any


class IdempotencyStore:
    def __init__(self) -> None:
        self._rows: Dict[str, Stored] = {}

    def get(self, request_id: str, fingerprint: str) -> Optional[Any]:
        row = self._rows.get(request_id)
        if row is None:
            return None
        if row.fingerprint != fingerprint:
            raise IdempotencyConflict(f"request_id {request_id} reused with a different payload")
        return row.result

    def put(self, request_id: str, fingerprint: str, result: Any) -> None:
        self._rows[request_id] = Stored(fingerprint, result)

    def __len__(self) -> int:
        return len(self._rows)
