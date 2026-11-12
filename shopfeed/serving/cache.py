"""Caches for the ranking path (spec §12.3). In-process TTL caches behind a small interface
so a Redis-backed implementation can replace them without touching callers."""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable
from typing import Any

from shopfeed import config


class TTLCache:
    def __init__(self, ttl_s: float, clock: Callable[[], float] = time.monotonic):
        self.ttl_s = ttl_s
        self.clock = clock
        self._data: dict[Hashable, tuple[float, Any]] = {}
        self.stats = {"hit": 0, "miss": 0}

    def get(self, key: Hashable) -> Any | None:
        item = self._data.get(key)
        if item is None or item[0] <= self.clock():
            self._data.pop(key, None)
            self.stats["miss"] += 1
            return None
        self.stats["hit"] += 1
        return item[1]

    def set(self, key: Hashable, value: Any) -> None:
        self._data[key] = (self.clock() + self.ttl_s, value)

    def delete(self, key: Hashable) -> None:
        self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()

    def keys(self):
        return list(self._data)

    def hit_rate(self) -> float:
        n = self.stats["hit"] + self.stats["miss"]
        return self.stats["hit"] / n if n else 0.0


class RankedListCache(TTLCache):
    """query_id -> {"item_ids": [...], "model_version": str}. Tied to one search session."""

    def __init__(self, ttl_s: float = config.RANKED_LIST_TTL_S, **kw):
        super().__init__(ttl_s, **kw)


class QueryResultCache(TTLCache):
    """(normalised query, device, aov tier, country, filters) -> ordered item_ids.

    Invalidated when any listed item's price or stock changes, and wholesale on model deploy.
    """

    def __init__(self, ttl_s: float = config.QUERY_CACHE_TTL_S, **kw):
        super().__init__(ttl_s, **kw)

    @staticmethod
    def key(query: str, device: str, aov_tier: str, country: str, filters: tuple = ()) -> tuple:
        return (" ".join(query.lower().split()), device, aov_tier, country, tuple(filters))

    def invalidate_items(self, item_ids) -> int:
        changed = set(item_ids)
        stale = [k for k, (_, v) in self._data.items() if changed & set(v)]
        for k in stale:
            self.delete(k)
        return len(stale)

    def on_model_deploy(self) -> None:
        self.clear()
