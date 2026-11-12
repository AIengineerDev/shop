"""Model feature matrix (spec §5.1). One code path for training and serving.

`build_features(rows, store)` takes CSV-shaped rows (one per candidate, grouped by qid)
and a HistoryStore, and returns (X, feature_names). It never reads labels,
displayed_pos or raw IDs (§5.2); `assert_no_excluded` enforces that on the output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shopfeed.features.history import HISTORY_FEATURES, HistoryStore

BASE_FEATURES = ["bm25", "semantic_sim", "price", "hist_ctr_7d", "rating"]
CATEGORICAL = {
    "user_aov_tier": {"Low": 0, "Mid": 1, "High": 2},
    "device": {"unknown": 0, "mobile": 1, "desktop": 2, "app": 3},
}
WITHIN_GROUP = ["bm25", "semantic_sim", "price", "hist_ctr_7d", "item_purchases_30d"]
EXCLUDED = {
    "displayed_pos", "clicked", "added_to_cart", "purchased", "wishlisted", "relevance_grade",
    "session_id", "qid", "user_id", "item_id", "seller_id", "query_text", "impression_ts",
}


def _within_group(x: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=x.index)
    g = x.groupby(group)
    for c in WITHIN_GROUP:
        col = x[c]
        out[f"{c}_rank_in_group"] = g[c].rank(ascending=False, method="min")
        mean, std = g[c].transform("mean"), g[c].transform("std")
        out[f"{c}_z_in_group"] = ((col - mean) / std.replace(0, np.nan)).fillna(0.0).where(
            col.notna())
        mx = g[c].transform("max")
        out[f"{c}_ratio_to_max"] = (col / mx.replace(0, np.nan)).where(col.notna())
    out["candidate_set_size"] = g["price"].transform("size")
    out["group_mean_semantic_sim"] = g["semantic_sim"].transform("mean")
    out["group_max_semantic_sim"] = g["semantic_sim"].transform("max")
    return out


def build_features(rows: pd.DataFrame, store: HistoryStore) -> tuple[pd.DataFrame, list[str]]:
    hist = store.features(rows)
    x = pd.concat([rows[BASE_FEATURES].astype(float), hist], axis=1)
    for col, vocab in CATEGORICAL.items():
        x[f"{col}_code"] = rows[col].map(vocab).astype(float)
    # Cross feature (§5.1 price fit): item price relative to the user's own point-in-time AOV.
    x["price_to_user_aov"] = x["price"] / x["user_aov_value"]
    x = pd.concat([x, _within_group(x, rows["qid"])], axis=1)
    names = list(x.columns)
    assert_no_excluded(names)
    return x.astype(float), names


def assert_no_excluded(names) -> None:
    leaked = sorted(set(names) & EXCLUDED)
    if leaked:
        raise ValueError(f"excluded columns in feature set (§5.2): {leaked}")


FEATURE_GROUPS = {
    "relevance": ["bm25", "semantic_sim"],
    "item": ["price", "hist_ctr_7d", "rating"],
    "history": HISTORY_FEATURES,
}
