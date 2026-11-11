"""Hash chain for price_changes rows."""

from __future__ import annotations

import hashlib
import json
from typing import Any

HASHED_FIELDS = (
    "change_id",
    "sku",
    "recommendation_id",
    "action_id",
    "old_price_cents",
    "new_price_cents",
    "source",
    "actor",
    "applied_at",
    "prev_change_id",
)


def row_hash(row: dict[str, Any]) -> str:
    payload = {k: row.get(k) for k in HASHED_FIELDS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
