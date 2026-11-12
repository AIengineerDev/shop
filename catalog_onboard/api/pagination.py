"""Opaque cursor pagination over (created_at, id)."""

from __future__ import annotations

import base64
import json

from fastapi import HTTPException

DEFAULT_LIMIT = 50
MAX_LIMIT = 500


def encode_cursor(created_at: str, row_id: int) -> str:
    return base64.urlsafe_b64encode(json.dumps([created_at, row_id]).encode()).decode().rstrip("=")


def decode_cursor(cursor: str | None) -> tuple[str, int] | None:
    if not cursor:
        return None
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        created_at, row_id = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        return str(created_at), int(row_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail={"detail": "invalid cursor", "code": "BAD_CURSOR"}) from exc


def clamp_limit(limit: int | None) -> int:
    if limit is None:
        return DEFAULT_LIMIT
    if limit < 1 or limit > MAX_LIMIT:
        raise HTTPException(
            status_code=422, detail={"detail": f"limit must be in [1, {MAX_LIMIT}]", "code": "BAD_LIMIT"}
        )
    return limit
