import numpy as np
import pandas as pd
import pytest

from shopfeed.config import DAY, RerankConfig
from shopfeed.rerank.policy import enforce_consecutive, rerank

NOW = 1_000_000.0


def _cands(n=60, n_merchants=5, n_categories=4, fresh_every=None, seed=0, merchant_of=None):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "product_id": [f"p{i}" for i in range(n)],
        "merchant_id": [merchant_of(i) if merchant_of else f"m{i % n_merchants}" for i in range(n)],
        "category": [f"c{i % n_categories}" for i in range(n)],
        "published_at": [NOW - (60 if fresh_every and i % fresh_every == 0 else 5 * DAY) for i in range(n)],
        "value": rng.random(n),
        "source": "two_tower",
    })
    return df


def _runs(page):
    best = cur = 0
    prev = None
    for m in page.merchant_id:
        cur = cur + 1 if m == prev else 1
        best = max(best, cur)
        prev = m
    return best


def test_page_size_subset_positions_reasons():
    cfg = RerankConfig()
    c = _cands()
    page = rerank(c, cfg, NOW)
    assert len(page) == cfg.page_size
    assert set(page.product_id) <= set(c.product_id) and page.product_id.is_unique
    assert list(page.position) == list(range(len(page)))
    assert page.reason.notna().all()


def test_no_run_of_same_merchant_longer_than_cap():
    cfg = RerankConfig(exploration_slots=0)
    # one merchant has the 15 highest values: greedy would stack it
    c = _cands(merchant_of=lambda i: "m_big" if i < 15 else f"m{i % 4}")
    c.loc[:14, "value"] = 10 + c.loc[:14, "value"]
    page = rerank(c, cfg, NOW)
    assert _runs(page) <= cfg.max_consecutive_merchant


def test_merchant_share_cap():
    cfg = RerankConfig(exploration_slots=0, merchant_share_cap=0.3)
    c = _cands(merchant_of=lambda i: "m_big" if i < 30 else f"m{i % 4}")
    c.loc[:29, "value"] = 10 + c.loc[:29, "value"]
    page = rerank(c, cfg, NOW)
    assert (page.merchant_id == "m_big").sum() <= int(0.3 * cfg.page_size)


def test_category_cap_in_top10():
    cfg = RerankConfig(exploration_slots=0, freshness_floor=0, max_same_category_top10=3)
    c = _cands(n_categories=4)
    c.loc[c.category == "c0", "value"] += 10  # one category dominates by value
    page = rerank(c, cfg, NOW)
    assert page.head(10).category.value_counts().max() <= 3 and len(page) == cfg.page_size


def test_category_cap_relaxed_only_when_page_would_be_short():
    cfg = RerankConfig(exploration_slots=0, freshness_floor=0, max_same_category_top10=3)
    page = rerank(_cands(n_categories=2), cfg, NOW)
    assert len(page) == cfg.page_size and (page.reason == "value_relaxed_category").any()


def test_merchant_cap_holds_with_fresh_floor_and_exploration():
    cfg = RerankConfig(exploration_slots=3, freshness_floor=2, merchant_share_cap=0.3)
    c = _cands(n=80, fresh_every=4, merchant_of=lambda i: "m_big" if i % 2 == 0 else f"m{i % 5}")
    c.loc[c.merchant_id == "m_big", "value"] += 10
    page = rerank(c, cfg, NOW, rng=np.random.default_rng(3))
    assert page.merchant_id.value_counts().max() <= int(0.3 * cfg.page_size)
    assert page.is_exploration.sum() == 3


def test_freshness_floor_met_when_fresh_exist_and_skipped_when_none():
    cfg = RerankConfig(exploration_slots=0, freshness_floor=2)
    c = _cands(fresh_every=7)  # fresh items spread over merchants, so the merchant cap does not block them
    c.loc[c.published_at > NOW - DAY, "value"] = -1.0  # fresh items have the worst value
    page = rerank(c, cfg, NOW)
    assert ((NOW - page.published_at) <= DAY).sum() >= 2
    assert (page.reason == "freshness_floor").sum() == 2
    page2 = rerank(_cands(fresh_every=None), cfg, NOW)
    assert ((NOW - page2.published_at) <= DAY).sum() == 0 and len(page2) == cfg.page_size


def test_freshness_floor_yields_to_merchant_cap():
    cfg = RerankConfig(exploration_slots=0, freshness_floor=2, merchant_share_cap=0.3)
    c = _cands(fresh_every=20)  # the 3 fresh items all belong to m0, which the greedy pass fills to its cap
    c.loc[c.published_at > NOW - DAY, "value"] = -1.0
    page = rerank(c, cfg, NOW)
    assert page.merchant_id.value_counts().max() <= int(0.3 * cfg.page_size)
    assert len(page) == cfg.page_size  # the floor never shortens the page


def test_exploration_slots_count_and_propensities():
    cfg = RerankConfig(exploration_slots=3)
    page = rerank(_cands(), cfg, NOW, rng=np.random.default_rng(1))
    expl = page[page.is_exploration]
    assert len(expl) == 3
    assert ((expl.propensity > 0) & (expl.propensity < 1)).all()
    assert (page[~page.is_exploration].propensity == 1.0).all()
    assert (expl.reason == "exploration").all()


def test_fewer_candidates_than_page_returns_all():
    cfg = RerankConfig()
    c = _cands(n=7)
    page = rerank(c, cfg, NOW)
    assert len(page) == 7 and set(page.product_id) == set(c.product_id)


def test_empty_candidates_and_missing_columns():
    cfg = RerankConfig()
    assert len(rerank(_cands(n=0), cfg, NOW)) == 0
    with pytest.raises(ValueError, match="missing columns"):
        rerank(pd.DataFrame({"product_id": ["a"]}), cfg, NOW)


def test_enforce_consecutive_swaps_then_drops():
    page = pd.DataFrame({"merchant_id": ["a", "a", "a", "b"], "product_id": list("wxyz")})
    out = enforce_consecutive(page, 2)
    assert list(out.merchant_id) == ["a", "a", "b", "a"]
    only_a = pd.DataFrame({"merchant_id": ["a", "a", "a"], "product_id": list("xyz")})
    assert len(enforce_consecutive(only_a, 2)) == 2
