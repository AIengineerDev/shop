"""Impression log -> leakage-safe training examples.

* One example per logged product impression whose label window has closed (ts + window <= now).
* Labels: events linked to that impression by impression_id, with impression_ts <= event_ts
  <= impression_ts + label_window_s. Later events are ignored: same window for everyone.
* Features: batch state refreshed daily (as of each day's start, strictly before the
  impressions it serves); streaming counters at the exact impression ts.
* Negatives are shown-but-unengaged impressions. Unseen items are never negatives.
* Time-ordered split with a gap >= label window so no validation label leaks into training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from shopfeed.config import DAY, EVENT_TO_HEAD, HEADS, POSITIVE_HEADS
from shopfeed.features.batch import compute_batch_state
from shopfeed.features.snapshot import N_FEATURES, build_features
from shopfeed.features.streaming import StreamingStore

_H = {h: i for i, h in enumerate(HEADS)}


@dataclass
class TrainingSet:
    X: np.ndarray
    Y: np.ndarray
    meta: pd.DataFrame  # impression_id, request_id, user_id, product_id, ts, position, propensity, source

    def __len__(self) -> int:
        return len(self.X)

    @property
    def any_positive(self) -> np.ndarray:
        idx = [_H[h] for h in HEADS if h in POSITIVE_HEADS]
        return (self.Y[:, idx].sum(axis=1) > 0).astype(int)


def build_labels(impressions: pd.DataFrame, events: pd.DataFrame, label_window_s: float) -> np.ndarray:
    Y = np.zeros((len(impressions), len(HEADS)), dtype=np.float32)
    if len(impressions) == 0:
        return Y
    ev = events[events.impression_id.notna() & events.type.isin(EVENT_TO_HEAD)]
    ev = ev.merge(impressions[["impression_id", "ts"]].rename(columns={"ts": "imp_ts"}), on="impression_id", how="inner")
    ev = ev[(ev.ts >= ev.imp_ts) & (ev.ts <= ev.imp_ts + label_window_s)]
    row = pd.Series(np.arange(len(impressions)), index=impressions.impression_id.values)
    r = row.reindex(ev.impression_id.values).values
    c = ev.type.map(EVENT_TO_HEAD).map(_H).values
    Y[r.astype(int), c.astype(int)] = 1.0
    return Y


def build_training_set(
    tables: Dict[str, pd.DataFrame],
    streaming: StreamingStore,
    now: float,
    label_window_s: float,
    refresh_s: float = DAY,
    max_rows: int = None,
) -> TrainingSet:
    imps = tables["impressions"]
    imps = imps[(imps.card_type == "product") & (imps.ts + label_window_s <= now)]
    imps = imps.sort_values("ts", kind="stable").reset_index(drop=True)
    if max_rows is not None and len(imps) > max_rows:
        imps = imps.sample(max_rows, random_state=0).sort_values("ts", kind="stable").reset_index(drop=True)
    if len(imps) == 0:
        raise ValueError("empty impression log: nothing to train on (no impression has a closed label window)")

    Y = build_labels(imps, tables["events"], label_window_s)
    X = np.zeros((len(imps), N_FEATURES), dtype=np.float32)
    t0 = np.floor(imps.ts.min() / refresh_s) * refresh_s
    bucket = ((imps.ts.values - t0) // refresh_s).astype(int)
    for b in np.unique(bucket):
        as_of = t0 + b * refresh_s  # batch features reflect only history strictly before this bucket
        rows = np.where(bucket == b)[0]
        state = compute_batch_state(tables["events"], tables["impressions"], tables["users"], tables["products"], tables["merchants"], as_of)
        X[rows] = build_features(imps.iloc[rows][["user_id", "product_id", "ts", "position"]], state, streaming, tables["products"], tables["follows"])
    meta = imps[["impression_id", "request_id", "user_id", "product_id", "ts", "position", "propensity", "source"]].reset_index(drop=True)
    return TrainingSet(X=X, Y=Y, meta=meta)


def time_split(meta: pd.DataFrame, val_fraction: float, gap_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Train = ts < cutoff - gap ; val = ts >= cutoff. Never random."""
    ts = meta.ts.values
    cutoff = np.quantile(ts, 1.0 - val_fraction)
    train = np.where(ts < cutoff - gap_s)[0]
    val = np.where(ts >= cutoff)[0]
    return train, val
