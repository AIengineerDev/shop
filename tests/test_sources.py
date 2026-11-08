import numpy as np
import pandas as pd
import pytest

from shopfeed.config import DAY, HOUR, EligibilityConfig, SourceCaps
from shopfeed.features.streaming import StreamingStore
from shopfeed.retrieval.ann import BruteForceIndex
from shopfeed.retrieval.sources import (UserContext, eligibility_mask, merge_candidates, source_complements,
                                        source_fresh, source_followed, source_replenish, source_trending, generate_candidates)

NOW = 1_000_000.0


def _products(rows):
    cols = ["product_id", "merchant_id", "category", "price", "in_stock", "ships_to", "published_at", "is_consumable",
            "repurchase_cycle_s", "is_evergreen", "merchant_status"]
    df = pd.DataFrame(rows, columns=cols)
    for d in range(8):
        df[f"c{d}"] = 0.1
    return df


BASE = ("m_a", "socks", 10.0, True, ["US", "CA"], NOW - 2 * DAY, False, np.nan, False, "active")


def _row(pid, **over):
    r = dict(zip(["merchant_id", "category", "price", "in_stock", "ships_to", "published_at", "is_consumable",
                  "repurchase_cycle_s", "is_evergreen", "merchant_status"], BASE))
    r.update(over)
    return (pid, *r.values())


@pytest.fixture
def catalog():
    return _products([
        _row("ok"),
        _row("oos", in_stock=False),
        _row("noship", ships_to=["DE"]),
        _row("susp", merchant_status="suspended"),
        _row("blocked", merchant_id="m_blocked"),
        _row("bought"),
        _row("shown"),
        _row("hidden"),
        _row("future", published_at=NOW + 10),
        _row("old", published_at=NOW - 40 * DAY),
        _row("old_evergreen", published_at=NOW - 40 * DAY, is_evergreen=True),
    ])


def test_eligibility_excludes_each_reason(catalog):
    ctx = UserContext(user_id="u", country="US", now=NOW, blocked_merchants={"m_blocked"}, recent_purchased={"bought"},
                      recent_shown={"shown"}, hidden={"hidden"})
    mask = eligibility_mask(catalog, ctx, EligibilityConfig())
    assert set(catalog.product_id[mask]) == {"ok", "old_evergreen"}


def test_fresh_source_cap_and_window(catalog):
    cat = catalog.copy()
    cat.loc[cat.product_id == "ok", "published_at"] = NOW - 60      # 1 minute old, zero interactions
    cat.loc[cat.product_id == "old_evergreen", "published_at"] = NOW - 2 * HOUR
    ctx = UserContext(user_id="u", country="US", now=NOW)
    mask = eligibility_mask(cat, ctx, EligibilityConfig())
    out = source_fresh(ctx, cat, mask, cap=1)
    assert len(out) == 1 and out.product_id.iloc[0] == "ok" and out.source.iloc[0] == "fresh"
    assert len(source_fresh(ctx, cat, mask, cap=10)) == 2


def test_merge_dedupes_keeping_max_score_and_lists_sources():
    a = pd.DataFrame({"product_id": ["x", "y"], "source": "two_tower", "score": [0.2, 0.9]})
    b = pd.DataFrame({"product_id": ["x"], "source": "fresh", "score": [0.7]})
    m = merge_candidates([a, b, pd.DataFrame(columns=["product_id", "source", "score"])])
    row = m.set_index("product_id").loc["x"]
    assert row.score == 0.7 and row.source == "fresh" and row.sources == ["fresh", "two_tower"]
    assert len(m) == 2


def test_complements_never_return_purchased_category(catalog):
    cat = _products([_row("shoes2", category="running_shoes"), _row("sock1", category="socks"), _row("insole1", category="insoles"), _row("mug", category="mugs")])
    ctx = UserContext(user_id="u", country="US", now=NOW, purchased_categories={"running_shoes"})
    mask = eligibility_mask(cat, ctx, EligibilityConfig())
    pop = pd.Series({"sock1": 5, "insole1": 1, "shoes2": 100})
    out = source_complements(ctx, cat, mask, pop, cap=10)
    assert set(out.product_id) == {"sock1", "insole1"}
    assert out.product_id.iloc[0] == "sock1"  # most popular first


def test_replenish_only_consumables_inside_window():
    cat = _products([
        _row("coffee_due", category="coffee", is_consumable=True, repurchase_cycle_s=30 * DAY),
        _row("coffee_fresh", category="coffee", is_consumable=True, repurchase_cycle_s=30 * DAY),
        _row("socks_old", category="socks"),
    ])
    ctx = UserContext(user_id="u", country="US", now=NOW,
                      purchase_last_ts={"coffee_due": NOW - 28 * DAY, "coffee_fresh": NOW - 3 * DAY, "socks_old": NOW - 28 * DAY},
                      recent_purchased={"coffee_fresh"})
    mask_no_repeat = eligibility_mask(cat, UserContext(user_id="u", country="US", now=NOW), EligibilityConfig(max_item_age_s=365 * DAY))
    out = source_replenish(ctx, cat, mask_no_repeat, cap=10)
    assert list(out.product_id) == ["coffee_due"]


def test_followed_source_empty_without_follows(catalog):
    ctx = UserContext(user_id="u", country="US", now=NOW)
    assert len(source_followed(ctx, catalog, eligibility_mask(catalog, ctx, EligibilityConfig()), 10)) == 0


def test_followed_source_excludes_blocked_followed_merchant(catalog):
    ctx = UserContext(user_id="u", country="US", now=NOW, followed_merchants={"m_blocked"}, blocked_merchants={"m_blocked"})
    mask = eligibility_mask(catalog, ctx, EligibilityConfig())
    assert len(source_followed(ctx, catalog, mask, 10)) == 0


def test_trending_empty_without_events(catalog):
    ctx = UserContext(user_id="u", country="US", now=NOW)
    out = source_trending(ctx, catalog, eligibility_mask(catalog, ctx, EligibilityConfig()), StreamingStore(), 10)
    assert len(out) == 0


def test_merchant_with_all_products_out_of_stock_never_surfaces():
    cat = _products([_row("a", merchant_id="m_dead", in_stock=False), _row("b", merchant_id="m_dead", in_stock=False), _row("c")])
    ctx = UserContext(user_id="u", country="US", now=NOW, followed_merchants={"m_dead"})
    idx = BruteForceIndex().build(cat.product_id.values, np.ones((3, 4), dtype=np.float32))
    ctx.user_vec = np.ones(4)
    out = generate_candidates(ctx, cat, idx, StreamingStore(), pd.Series(dtype=float), SourceCaps(), EligibilityConfig())
    assert "m_dead" not in set(cat.set_index("product_id").loc[out.product_id].merchant_id)


def test_generate_candidates_respects_caps(world, trained, batch, streaming):
    from shopfeed.retrieval.sources import enrich_products
    prods = enrich_products(world["products"], world["merchants"])
    idx = BruteForceIndex().build(trained["retrieval"].item_ids, trained["retrieval"].item_vecs)
    u = world["users"].iloc[0]
    ctx = UserContext(user_id=u.user_id, country=u.country, now=1_750_000_000.0, user_vec=trained["retrieval"].embed_users(batch.users.head(1))[0])
    caps = SourceCaps(two_tower=7, fresh=1, trending=2)
    out = generate_candidates(ctx, prods, idx, streaming, batch.products.set_index("product_id").n_purchases, caps, EligibilityConfig())
    counts = out.explode("sources").groupby("sources").size()
    assert counts.get("two_tower", 0) <= 7 and counts.get("fresh", 0) <= 1 and counts.get("trending", 0) <= 2
    assert out.product_id.is_unique
