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


def _group_index(group) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(pd.Series(group).to_numpy())
    return codes, len(uniques)


def _rank_desc_min(x: np.ndarray, g: np.ndarray, n_groups: int) -> np.ndarray:
    """Per-group rank, descending, ties -> minimum rank, NaN stays NaN (pandas
    groupby().rank(ascending=False, method="min"))."""
    out = np.full(len(x), np.nan)
    ok = ~np.isnan(x)
    if not ok.any():
        return out
    idx = np.flatnonzero(ok)
    order = idx[np.lexsort((-x[idx], g[idx]))]
    gs, xs = g[order], x[order]
    pos = np.arange(len(order))
    new_group = np.r_[True, gs[1:] != gs[:-1]]
    group_start = np.maximum.accumulate(np.where(new_group, pos, 0))
    new_value = new_group | np.r_[True, xs[1:] != xs[:-1]]
    tie_start = np.maximum.accumulate(np.where(new_value, pos, 0))
    out[order] = tie_start - group_start + 1
    return out


def _group_stats(x: np.ndarray, g: np.ndarray, n: int):
    ok = ~np.isnan(x)
    cnt = np.bincount(g, weights=ok, minlength=n)
    s = np.bincount(g, weights=np.where(ok, x, 0.0), minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, s / cnt, np.nan)
        dev = np.where(ok, x - mean[g], 0.0)
        ss = np.bincount(g, weights=dev * dev, minlength=n)
        std = np.where(cnt > 1, np.sqrt(ss / (cnt - 1)), np.nan)
    mx = np.full(n, -np.inf)
    np.maximum.at(mx, g[ok], x[ok])
    mx = np.where(cnt > 0, mx, np.nan)
    return mean, std, mx


def _within_group(x: pd.DataFrame, group) -> pd.DataFrame:
    """Within-query rank / z-score / ratio-to-max (§5.1), vectorised with bincount so the
    same code is fast for one 200-row request and for the full training set."""
    g, n = _group_index(group)
    out: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for c in WITHIN_GROUP:
            col = x[c].to_numpy(dtype=float)
            mean, std, mx = _group_stats(col, g, n)
            ok = ~np.isnan(col)
            out[f"{c}_rank_in_group"] = _rank_desc_min(col, g, n)
            sd = std[g]
            z = np.where((sd > 0) & ~np.isnan(sd), (col - mean[g]) / sd, 0.0)
            out[f"{c}_z_in_group"] = np.where(ok, z, np.nan)
            m = mx[g]
            out[f"{c}_ratio_to_max"] = np.where(ok & (m != 0), col / m, np.nan)
        out["candidate_set_size"] = np.bincount(g, minlength=n)[g].astype(float)
        sem_mean, _, sem_max = _group_stats(x["semantic_sim"].to_numpy(dtype=float), g, n)
        out["group_mean_semantic_sim"] = sem_mean[g]
        out["group_max_semantic_sim"] = sem_max[g]
    return pd.DataFrame(out, index=x.index)


def build_features(rows: pd.DataFrame, store: HistoryStore) -> tuple[pd.DataFrame, list[str]]:
    hist = store.features(rows)
    x = pd.concat([rows[BASE_FEATURES].astype(float), hist], axis=1)
    extra = {f"{col}_code": rows[col].map(vocab).astype(float).to_numpy()
             for col, vocab in CATEGORICAL.items()}
    # Cross feature (§5.1 price fit): item price relative to the user's own point-in-time AOV.
    extra["price_to_user_aov"] = (x["price"] / x["user_aov_value"]).to_numpy()
    x = pd.concat([x, pd.DataFrame(extra, index=x.index)], axis=1)
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
