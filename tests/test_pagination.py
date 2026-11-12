"""Rank-once pagination and caches (spec §12.2-12.3, §13.1)."""

from shopfeed.serving.cache import QueryResultCache, RankedListCache, TTLCache
from shopfeed.serving.pagination import Paginator, decode_cursor, encode_cursor

ITEMS = [f"p_{i}" for i in range(200)]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _pages(p: Paginator, qid=7):
    page = p.first_page(qid)
    pages = [page]
    while page.next_cursor:
        page = p.next_page(page.next_cursor)
        pages.append(page)
    return pages


def test_200_items_paged_by_20_are_unique_and_ordered():
    calls = []
    p = Paginator(lambda q: calls.append(q) or list(ITEMS), page_size=20)
    pages = _pages(p)
    served = [i for pg in pages for i in pg.item_ids]
    assert served == ITEMS and len(pages) == 10
    assert pages[2].positions[0] == 41  # absolute positions
    assert calls == [7]  # ranked once
    assert all(pg.from_cache for pg in pages[1:])


def test_cache_miss_on_page_3_does_not_repeat_items():
    clock = Clock()
    rerank_order = list(reversed(ITEMS))  # a re-rank produces a different order
    calls = {"n": 0}

    def rank(q):
        calls["n"] += 1
        return list(ITEMS) if calls["n"] == 1 else rerank_order

    p = Paginator(rank, RankedListCache(ttl_s=10, clock=clock), page_size=20)
    p1 = p.first_page(1)
    p2 = p.next_page(p1.next_cursor)
    clock.t = 100  # expire
    p3 = p.next_page(p2.next_cursor)
    first_two = set(p1.item_ids + p2.item_ids)
    assert not first_two & set(p3.item_ids)
    assert p3.positions[0] == 41 and p.counters["cache_miss_rerank"] == 1
    rest = [i for pg in [p3] for i in pg.item_ids]
    nxt = p3
    while nxt.next_cursor:
        nxt = p.next_page(nxt.next_cursor)
        rest += nxt.item_ids
    assert len(set(p1.item_ids + p2.item_ids + rest)) == 200


def test_cursor_roundtrip():
    assert decode_cursor(encode_cursor(123, 40)) == (123, 40)


def test_ttl_cache_expiry_and_hit_rate():
    clock = Clock()
    c = TTLCache(5, clock=clock)
    c.set("k", 1)
    assert c.get("k") == 1
    clock.t = 6
    assert c.get("k") is None
    assert c.hit_rate() == 0.5


def test_query_cache_invalidation_on_price_change_and_deploy():
    c = QueryResultCache()
    k1 = QueryResultCache.key("  Red  Mugs ", "mobile", "High", "US")
    k2 = QueryResultCache.key("socks", "mobile", "High", "US")
    assert k1 == QueryResultCache.key("red mugs", "mobile", "High", "US")
    c.set(k1, ["p_1", "p_2"])
    c.set(k2, ["p_9"])
    assert c.invalidate_items(["p_2"]) == 1
    assert c.get(k1) is None and c.get(k2) == ["p_9"]
    c.on_model_deploy()
    assert c.get(k2) is None
