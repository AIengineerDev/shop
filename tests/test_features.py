"""Point-in-time history features, best-seller rank, within-group features, §5.2 exclusions."""

import numpy as np
import pandas as pd
import pytest

from shopfeed import config
from shopfeed.features.history import HISTORY_FEATURES, HistoryStore
from shopfeed.features.ranking import EXCLUDED, assert_no_excluded, build_features
from shopfeed.schema import read_csv

DAY = 86400.0
T = 100 * DAY  # impression time used throughout


def _events(rows):
    cols = ["event_id", "user_id", "product_id", "merchant_id", "type", "ts", "impression_id",
            "value"]
    return pd.DataFrame(rows, columns=cols)


def _store(events, n_items=5, days=None):
    products = pd.DataFrame({
        "product_id": [f"p_{i}" for i in range(n_items)],
        "merchant_id": ["m_1"] * (n_items - 1) + ["m_2"],
        "category": ["mugs"] * n_items,
        "published_at": [T - 10 * DAY] * (n_items - 1) + [T + DAY],  # last one is "future"
    })
    merchants = pd.DataFrame({"merchant_id": ["m_1", "m_2"], "return_rate": [0.1, 0.3],
                              "fulfilment_days": [1.0, 3.0]})
    imps = pd.DataFrame({"product_id": ["p_0"], "ts": [T - DAY]})
    return HistoryStore.from_sources(imps, events, products, merchants,
                                     days=days or range(int(T // DAY), int(T // DAY) + 1))


def _rows(items, user="u_1", seller="m_1", t=T):
    return pd.DataFrame({"item_id": items, "seller_id": seller, "user_id": user,
                         "impression_ts": t})


def _purchases(spec):
    """spec: list of (item, n, ts)."""
    rows, k = [], 0
    for item, n, ts in spec:
        for _ in range(n):
            rows.append((f"e{k}", "u_9", item, "m_1", "purchase", ts, None, 10.0))
            k += 1
    return rows


def test_history_counts_are_point_in_time():
    ev = _events(_purchases([("p_0", 2, T - 5 * DAY), ("p_0", 1, T - 40 * DAY)]))
    f = _store(ev).features(_rows(["p_0"]))
    assert f["item_purchases_30d"].item() == 2
    assert f["item_purchases_all"].item() == 3
    future = _events(_purchases([("p_0", 2, T - 5 * DAY), ("p_0", 1, T - 40 * DAY),
                                 ("p_0", 5, T + 1), ("p_0", 1, T)]))
    pd.testing.assert_frame_equal(f, _store(future).features(_rows(["p_0"])))


def test_bestseller_rank_in_category():
    ev = _events(_purchases([("p_3", 4, T - DAY), ("p_1", 2, T - DAY), ("p_2", 2, T - DAY)]))
    f = _store(ev).features(_rows(["p_0", "p_1", "p_2", "p_3", "p_4"]))
    # p_3 best; p_1 / p_2 tie broken by id; p_0 / p_4 have no sales.
    assert list(f["item_bestseller_rank_category"]) == [4, 2, 3, 1, 5]
    assert list(f["is_category_top20"]) == [0, 1, 1, 1, 0]


def test_popularity_uses_day_start():
    same_day = _events(_purchases([("p_1", 3, T + 3600)]))  # after the day started
    f = _store(same_day).features(_rows(["p_1"], t=T + 7200))
    assert f["item_bestseller_rank_category"].item() > 1
    assert f["item_purchases_30d"].item() == 3  # exact-t counts do see it


def test_seller_user_and_age_features():
    ev = _events(_purchases([("p_1", 1, T - DAY)]) + [
        ("f1", "u_1", None, "m_1", "follow", T - DAY, None, np.nan),
        ("f2", "u_1", None, "m_2", "follow", T + DAY, None, np.nan),  # future follow
        ("b1", "u_1", "p_2", "m_1", "purchase", T - 2 * DAY, None, 40.0),
        ("b2", "u_1", "p_3", "m_1", "purchase", T - 3 * DAY, None, 60.0),
    ])
    s = _store(ev)
    f = s.features(pd.DataFrame({"item_id": ["p_0", "p_4"], "seller_id": ["m_1", "m_2"],
                                 "user_id": "u_1", "impression_ts": T}))
    assert list(f["user_follows_seller"]) == [1, 0]
    assert list(f["seller_item_count"]) == [4, 0]  # p_4 published after T
    assert list(f["seller_return_rate"]) == [0.1, 0.3]
    assert f["item_age_days"].iloc[0] == pytest.approx(10.0)
    assert f["item_age_days"].iloc[1] == 0.0  # clipped, not negative
    assert f["user_order_count"].iloc[0] == 2 and f["is_new_user"].iloc[0] == 0
    assert f["user_aov_value"].iloc[0] == pytest.approx(50.0)
    new = s.features(_rows(["p_0"], user="u_new"))
    assert new["is_new_user"].item() == 1 and np.isnan(new["user_aov_value"].item())


def test_build_features_on_fixture():
    df = read_csv(config.FIXTURE_CSV)
    X, names = build_features(df, _store(_events([])))
    assert len(X) == 10 and names == list(X.columns)
    assert not set(names) & EXCLUDED
    assert set(HISTORY_FEATURES) <= set(names)
    g1 = X[df["qid"] == 1]
    # bm25 in group 1: 14.8, 12.1, 13.5, 10.9, 8.2 -> ranks 1, 3, 2, 4, 5
    assert list(g1["bm25_rank_in_group"]) == [1, 3, 2, 4, 5]
    assert g1["bm25_ratio_to_max"].iloc[0] == pytest.approx(1.0)
    assert g1["bm25_z_in_group"].mean() == pytest.approx(0.0, abs=1e-9)
    assert (X["candidate_set_size"] == 5).all()
    assert list(X["user_aov_tier_code"].unique()) == [2.0, 1.0]


def test_train_serve_parity_single_group():
    """Scoring one group alone (serving) gives the same features as inside the full frame."""
    df = read_csv(config.FIXTURE_CSV)
    store = _store(_events([]))
    full, _ = build_features(df, store)
    alone, _ = build_features(df[df["qid"] == 2].reset_index(drop=True), store)
    pd.testing.assert_frame_equal(full[df["qid"] == 2].reset_index(drop=True), alone)


def test_excluded_feature_guard():
    with pytest.raises(ValueError, match="displayed_pos"):
        assert_no_excluded(["semantic_sim", "displayed_pos"])
