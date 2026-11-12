"""Cursor pagination over a rank-once list (spec §12.2).

Page 1 ranks the whole candidate set once and caches the ordered list under query_id.
Pages 2+ are slices of the cached list. On a cache miss (expired / evicted) the request is
re-ranked, every item already served for this query_id is removed, and paging continues;
the miss is counted. Positions are absolute across pages so propensities stay comparable.
"""

from __future__ import annotations

import base64
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from shopfeed.serving.cache import RankedListCache


def encode_cursor(qid: int, offset: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"q": qid, "o": offset}).encode()).decode()


def decode_cursor(cursor: str) -> tuple[int, int]:
    d = json.loads(base64.urlsafe_b64decode(cursor.encode()))
    return int(d["q"]), int(d["o"])


@dataclass
class Page:
    item_ids: list[str]
    positions: list[int]  # absolute, 1-based
    next_cursor: str | None
    from_cache: bool


class Paginator:
    def __init__(self, rank_fn: Callable[[int], list[str]], cache: RankedListCache | None = None,
                 page_size: int = 20):
        """rank_fn(qid) -> full ordered item_ids for that query (the ranker call)."""
        self.rank_fn = rank_fn
        self.cache = cache or RankedListCache()
        self.page_size = page_size
        self.served: dict[int, list[str]] = defaultdict(list)  # stands in for item_impressed
        self.counters = {"cache_miss_rerank": 0}

    def first_page(self, qid: int) -> Page:
        ids = self.rank_fn(qid)
        self.cache.set(qid, {"item_ids": ids})
        self.served[qid] = []
        return self._slice(qid, ids, 0, from_cache=False)

    def next_page(self, cursor: str) -> Page:
        qid, offset = decode_cursor(cursor)
        hit = self.cache.get(qid)
        if hit is not None:
            return self._slice(qid, hit["item_ids"], offset, from_cache=True)
        self.counters["cache_miss_rerank"] += 1
        seen = set(self.served[qid])
        fresh = [i for i in self.rank_fn(qid) if i not in seen]
        # Rebuild a list whose first `offset` slots are what was already served.
        ids = self.served[qid][:offset] + fresh
        self.cache.set(qid, {"item_ids": ids})
        return self._slice(qid, ids, offset, from_cache=False)

    def _slice(self, qid: int, ids: list[str], offset: int, from_cache: bool) -> Page:
        page = ids[offset: offset + self.page_size]
        self.served[qid].extend(page)
        end = offset + len(page)
        nxt = encode_cursor(qid, end) if end < len(ids) else None
        return Page(page, list(range(offset + 1, end + 1)), nxt, from_cache)
