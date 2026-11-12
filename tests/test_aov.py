import numpy as np
import pandas as pd

from shopfeed import config
from shopfeed.data.aov import compute_aov, daily_tiers

DAY = 86400.0


def _purchases(values, ts=0.0):
    return pd.DataFrame({
        "event_id": [f"e{i}" for i in range(len(values))],
        "user_id": [f"u_{i}" for i in range(len(values))],
        "product_id": [f"p_{i}" for i in range(len(values))],
        "type": "purchase", "ts": ts, "value": values,
    })


def test_tiers_split_buyers_into_thirds():
    ev = _purchases(list(np.arange(1, 31, dtype=float)))
    aov, _ = compute_aov(ev, asof_ts=DAY)
    counts = aov["aov_tier"].value_counts()
    assert all(abs(counts[t] - 10) <= 1 for t in config.AOV_TIERS)


def test_user_without_orders_is_mid():
    ev = _purchases([10.0, 50.0, 90.0])
    aov, _ = compute_aov(ev, asof_ts=DAY, user_ids=pd.Index(["u_0", "u_new"]))
    new = aov.set_index("user_id").loc["u_new"]
    assert new["aov_tier"] == "Mid" and new["aov_order_count"] == 0


def test_future_purchase_does_not_change_tier():
    ev = _purchases([10.0, 50.0, 90.0])
    before, _ = compute_aov(ev, asof_ts=DAY)
    future = pd.concat([ev, _purchases([1000.0], ts=2 * DAY).assign(user_id="u_0",
                                                                    event_id="ef")])
    after, _ = compute_aov(future, asof_ts=DAY)
    pd.testing.assert_frame_equal(before, after)


def test_returned_purchase_excluded():
    ev = _purchases([10.0, 50.0, 90.0])
    ret = pd.DataFrame({"event_id": ["r1"], "user_id": ["u_2"], "product_id": ["p_2"],
                        "type": ["return"], "ts": [DAY / 2], "value": [np.nan]})
    aov, _ = compute_aov(pd.concat([ev, ret]), asof_ts=DAY)
    assert "u_2" not in set(aov["user_id"])


def test_lookback_window():
    ev = _purchases([10.0], ts=0.0)
    aov, _ = compute_aov(ev, asof_ts=config.AOV_LOOKBACK_S + DAY)
    assert aov.empty


def test_rerun_is_deterministic():
    ev = _purchases(list(np.random.default_rng(0).uniform(5, 500, 100)))
    a = compute_aov(ev, DAY)
    b = compute_aov(ev, DAY)
    pd.testing.assert_frame_equal(a[0], b[0])
    assert a[1] == b[1]


def test_daily_tiers_are_as_of_day_start():
    ev = _purchases([10.0, 50.0, 90.0], ts=DAY + 100)
    d = daily_tiers(ev, ["u_0", "u_2"], start_ts=DAY + 50, end_ts=2 * DAY + 50)
    first = d[d["day"] == 1].set_index("user_id")["aov_tier"]
    second = d[d["day"] == 2].set_index("user_id")["aov_tier"]
    assert (first == "Mid").all()  # purchases happen after day 1 starts
    assert second["u_0"] == "Low" and second["u_2"] == "High"
