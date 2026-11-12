"""Temporal 60/20/20 session-level split (spec §7.1).

Sessions are ordered by their first impression_ts. Cut points are chosen so the shares
are measured in sessions. A session never straddles partitions, which implies qid
integrity. Boundary sessions whose impressions overlap the next partition's start are
purged (dropped from both sides) so that max(train ts) < min(val ts) < ... strictly holds;
the purge count goes to meta.json. Users seen in train are flagged for the cold-start slice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

SHARES = (0.6, 0.2, 0.2)
PARTITIONS = ("train", "val", "test")


@dataclass
class SplitMeta:
    cut_train_val_ts: float
    cut_val_test_ts: float
    sessions: dict
    rows: dict
    purged_sessions: int

    def to_dict(self) -> dict:
        return asdict(self)


def temporal_split(df: pd.DataFrame, shares=SHARES) -> tuple[pd.Series, SplitMeta]:
    """Return (partition label per row, or <NA> for purged rows; SplitMeta)."""
    sess = df.groupby("session_id")["impression_ts"].agg(["min", "max"]).sort_values("min")
    n = len(sess)
    i1, i2 = round(n * shares[0]), round(n * (shares[0] + shares[1]))
    part = np.array(["train"] * i1 + ["val"] * (i2 - i1) + ["test"] * (n - i2), dtype=object)
    sess["part"] = part

    cut1 = float(sess["min"].iloc[i1]) if i1 < n else float("inf")
    cut2 = float(sess["min"].iloc[i2]) if i2 < n else float("inf")
    # Purge sessions that end after the next partition starts.
    overlap = ((sess["part"] == "train") & (sess["max"] >= cut1)) | (
        (sess["part"] == "val") & (sess["max"] >= cut2))
    sess.loc[overlap, "part"] = None

    label = df["session_id"].map(sess["part"]).astype("string")
    meta = SplitMeta(
        cut_train_val_ts=cut1, cut_val_test_ts=cut2,
        sessions={p: int((sess["part"] == p).sum()) for p in PARTITIONS},
        rows={p: int((label == p).sum()) for p in PARTITIONS},
        purged_sessions=int(overlap.sum()),
    )
    return label, meta


def seen_user_flag(df: pd.DataFrame, partition: pd.Series) -> pd.Series:
    train_users = set(df.loc[partition == "train", "user_id"])
    return df["user_id"].isin(train_users).astype("int64").rename("user_seen_in_train")
