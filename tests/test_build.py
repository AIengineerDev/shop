"""Parquet -> CSV build: sessionization, attribution, dedup, labels, point-in-time CTR."""

import numpy as np
import pandas as pd
import pytest

from shopfeed.data.build import (
    BuildReport,
    Sources,
    attribute_events,
    build_rows,
    hist_ctr_7d,
    sessionize,
)
from shopfeed.labels import consistency_violations
from shopfeed.schema import validate

T0 = 1_746_000_000.0
H = 3600.0


def _imp(rows):
    cols = ["impression_id", "request_id", "user_id", "product_id", "ts", "position"]
    df = pd.DataFrame(rows, columns=cols)
    return df.assign(card_type="product", propensity=0.01, model_version="v0",
                     source="logging_random")


def _ev(rows):
    cols = ["event_id", "user_id", "product_id", "type", "ts", "impression_id", "value"]
    return pd.DataFrame(rows, columns=cols).assign(merchant_id="m_0")


@pytest.fixture
def src():
    imp = _imp([
        ("i1", "req_1", "u_1", "p_1", T0, 0),
        ("i2", "req_1", "u_1", "p_2", T0, 1),
        ("i3", "req_1", "u_1", "p_3", T0, 2),
        ("i4", "req_1", "u_1", "p_2", T0, 3),          # duplicate of p_2 in req_1
        ("i5", "req_2", "u_1", "p_1", T0 + 10 * 60, 0),  # same session (10 min later)
        ("i6", "req_2", "u_1", "p_4", T0 + 10 * 60, 1),
        ("i7", "req_3", "u_1", "p_1", T0 + 3 * H, 0),    # new session (gap > 30 min)
        ("i8", "req_3", "u_1", "p_3", T0 + 3 * H, 1),
    ])
    ev = _ev([
        ("e1", "u_1", "p_1", "view", T0 + 5, "i1", np.nan),
        ("e2", "u_1", "p_1", "purchase", T0 + 30 * H, "i1", 50.0),  # stamped, lag > 24 h
        ("e3", "u_1", "p_2", "save", T0 + 20, "i4", np.nan),         # on the duplicate
        ("e4", "u_1", "p_3", "view", T0 + 5, "i3", np.nan),
        ("e5", "u_1", "p_3", "hide", T0 + 9, "i3", np.nan),           # click-only -> 0
        ("e6", "u_1", "p_4", "add_to_cart", T0 + 11 * 60, None, np.nan),  # unstamped
        ("e7", "u_1", "p_9", "purchase", T0, None, 10.0),            # never impressed
    ])
    prod = pd.DataFrame({
        "product_id": ["p_1", "p_2", "p_3", "p_4", "p_9"],
        "merchant_id": ["m_1", "m_2", "m_1", "m_3", "m_3"],
        "category": ["mugs", "socks", "mugs", "coffee", "coffee"],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0],
        **{f"c{i}": [1.0, 0.0, -1.0, 0.5, 0.2] for i in range(8)},
    })
    users = pd.DataFrame({"user_id": ["u_1"], **{f"t{i}": [1.0] for i in range(8)}})
    return Sources(imp, ev, prod, users)


def test_sessionize_uses_30_min_gap(src):
    s = sessionize(src.impressions)
    assert s["req_1"] == s["req_2"]
    assert s["req_1"] != s["req_3"]


def test_attribution_stamped_and_reconstructed(src):
    r = BuildReport()
    a = attribute_events(src.impressions, src.events, r)
    assert set(a.loc[a["type"] == "purchase", "impression_id"]) == {"i1"}  # lag 30 h trusted
    assert set(a.loc[a["type"] == "add_to_cart", "impression_id"]) == {"i6"}  # reconstructed
    assert r.counts["purchases_stamped_lag_gt_24h"] == 1
    assert r.counts["events_unstamped_reconstructed"] == 1  # p_9 purchase is non-search


def test_build_rows(src):
    df = build_rows(src)
    validate(df)
    assert len(df) == 7  # 8 impressions, one duplicate merged
    r1 = df[df["qid"] == 1].set_index("item_id")
    assert list(r1["displayed_pos"]) == [1, 2, 3]  # duplicate p_2 kept at earliest position
    assert r1.loc["p_1", "relevance_grade"] == 4
    assert r1.loc["p_2", "relevance_grade"] == 2 and r1.loc["p_2", "clicked"] == 1
    assert r1.loc["p_3", "relevance_grade"] == 0 and r1.loc["p_3", "clicked"] == 1
    assert df[(df["qid"] == 2) & (df["item_id"] == "p_4")]["relevance_grade"].item() == 3
    assert consistency_violations(df).sum() == 0
    assert df["device"].eq("unknown").all() and df["bm25"].isna().all()
    assert df.loc[df["item_id"] == "p_1", "seller_id"].iloc[0] == "m_1"
    s = df.groupby("qid")["session_id"].first()
    assert s[1] == s[2] != s[3]


def test_semantic_sim_is_cosine(src):
    df = build_rows(src)
    sim = df.groupby("item_id")["semantic_sim"].first()
    assert sim["p_1"] == pytest.approx(1.0)
    assert sim["p_3"] == pytest.approx(-1.0)


def test_hist_ctr_is_point_in_time():
    imp = pd.DataFrame({"product_id": ["p_1"] * 4, "ts": [0.0, 10.0, 20.0, 30.0]})
    clicks = pd.DataFrame({"product_id": ["p_1"] * 2, "ts": [5.0, 25.0]})
    rows = pd.DataFrame({"item_id": ["p_1", "p_1"], "impression_ts": [20.0, 0.0]})
    ctr = hist_ctr_7d(rows, imp, clicks)
    assert ctr.iloc[0] == pytest.approx(0.5)  # 1 click / 2 impressions strictly before t=20
    assert np.isnan(ctr.iloc[1])  # nothing before t=0
    later = pd.concat([clicks, pd.DataFrame({"product_id": ["p_1"], "ts": [19.0]})])
    later_imp = pd.concat([imp, pd.DataFrame({"product_id": ["p_1"], "ts": [99.0]})])
    assert hist_ctr_7d(rows, later_imp, later).iloc[0] == pytest.approx(1.0)
    future_only = pd.concat([clicks, pd.DataFrame({"product_id": ["p_1"], "ts": [21.0]})])
    assert hist_ctr_7d(rows, imp, future_only).iloc[0] == pytest.approx(0.5)
