"""User Average Order Value tier (spec §2 `user_aov_tier`).

`compute_aov(events, asof_ts)` uses only purchases strictly before `asof_ts` within the
trailing AOV_LOOKBACK_S window, excluding purchases returned before `asof_ts`. Tier cut
points are the 33rd / 67th percentiles of aov_value over users with >= 1 order at that
instant; users with no orders get AOV_DEFAULT_TIER.

Training rows use `daily_tiers` (point-in-time at day granularity). The snapshot written
into users.parquet by scripts/build_user_aov.py is present-day and is for serving only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shopfeed import config

DAY_S = 86400


def _kept_purchases(events: pd.DataFrame, asof_ts: float) -> pd.DataFrame:
    lo = asof_ts - config.AOV_LOOKBACK_S
    p = events[(events["type"] == "purchase") & (events["ts"] < asof_ts) & (events["ts"] >= lo)]
    r = events[(events["type"] == "return") & (events["ts"] < asof_ts)]
    if len(r):
        m = p.merge(r[["user_id", "product_id", "ts"]], on=["user_id", "product_id"],
                    how="left", suffixes=("", "_ret"))
        returned = m["ts_ret"].notna() & (m["ts_ret"] > m["ts"])
        returned_ids = set(m.loc[returned, "event_id"])
        p = p[~p["event_id"].isin(returned_ids)]
    return p[p["value"].notna()]


def cut_points(aov_values: pd.Series) -> tuple[float, float]:
    if aov_values.empty:
        return (np.nan, np.nan)
    lo, hi = np.quantile(aov_values.to_numpy(), [1 / 3, 2 / 3])
    return float(lo), float(hi)


def assign_tier(aov_value: pd.Series, cuts: tuple[float, float]) -> pd.Series:
    lo, hi = cuts
    tier = np.where(aov_value <= lo, "Low", np.where(aov_value <= hi, "Mid", "High"))
    tier = np.where(aov_value.isna(), config.AOV_DEFAULT_TIER, tier)
    return pd.Series(tier, index=aov_value.index, dtype="string")


def compute_aov(
    events: pd.DataFrame, asof_ts: float, user_ids: pd.Index | None = None
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Return (frame[user_id, aov_value, aov_order_count, aov_tier], cut_points)."""
    p = _kept_purchases(events, asof_ts)
    agg = p.groupby("user_id")["value"].agg(aov_value="mean", aov_order_count="size")
    cuts = cut_points(agg["aov_value"])
    if user_ids is not None:
        agg = agg.reindex(pd.Index(user_ids, name="user_id"))
    agg["aov_order_count"] = agg["aov_order_count"].fillna(0).astype("int64")
    agg["aov_tier"] = assign_tier(agg["aov_value"], cuts)
    return agg.reset_index(), cuts


def daily_tiers(events: pd.DataFrame, user_ids, start_ts: float, end_ts: float) -> pd.DataFrame:
    """Tier per (user_id, day) where day = floor(ts / DAY_S), as of the day's start."""
    first, last = int(start_ts // DAY_S), int(end_ts // DAY_S)
    frames = []
    for day in range(first, last + 1):
        frame, _ = compute_aov(events, float(day * DAY_S), pd.Index(user_ids))
        frame["day"] = day
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
