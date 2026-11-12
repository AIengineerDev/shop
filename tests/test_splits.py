"""Temporal session-level split (spec §7.1, §13.1)."""

import numpy as np
import pandas as pd

from shopfeed.training.splits import seen_user_flag, temporal_split


def _sessions(n=500, rows_per=4, seed=0):
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.uniform(0, 1e6, n))
    rows = []
    for i, s in enumerate(starts):
        for j in range(rows_per):
            rows.append({"session_id": f"s_{i}", "qid": i, "user_id": f"u_{i % 50}",
                         "impression_ts": s + j})
    return pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)


def test_shares_within_one_point():
    df = _sessions()
    part, meta = temporal_split(df)
    n = sum(meta.sessions.values())
    for p, share in zip(("train", "val", "test"), (0.6, 0.2, 0.2), strict=True):
        assert abs(meta.sessions[p] / n - share) <= 0.01


def test_no_group_or_session_straddles():
    df = _sessions()
    df["part"], _ = temporal_split(df)
    assert (df.groupby("session_id")["part"].nunique() == 1).all()
    assert (df.groupby("qid")["part"].nunique() == 1).all()


def test_strict_time_order():
    df = _sessions()
    df["part"], meta = temporal_split(df)
    t = df.groupby("part")["impression_ts"].agg(["min", "max"])
    assert t.loc["train", "max"] < t.loc["val", "min"]
    assert t.loc["val", "max"] < t.loc["test", "min"]
    assert meta.cut_train_val_ts == t.loc["val", "min"]


def test_long_boundary_session_is_purged():
    df = _sessions(n=10, rows_per=1)
    order = df.sort_values("impression_ts")
    first = order["session_id"].iloc[0]
    late = order["impression_ts"].iloc[-1] + 1
    df = pd.concat([df, pd.DataFrame([{"session_id": first, "qid": 999, "user_id": "u_x",
                                       "impression_ts": late}])], ignore_index=True)
    part, meta = temporal_split(df)
    assert meta.purged_sessions == 1
    assert part[df["session_id"] == first].isna().all()


def test_seen_user_flag():
    df = pd.DataFrame({"user_id": ["a", "b", "a", "c"]})
    part = pd.Series(["train", "train", "test", "test"])
    assert list(seen_user_flag(df, part)) == [1, 1, 1, 0]
