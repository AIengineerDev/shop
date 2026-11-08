"""The single place a serving-time feature vector is assembled.

Training and serving both call `build_features`, so the two cannot diverge. The returned
matrix is what gets logged next to each impression. Position is the LAST column so the
ranker can zero it at inference (see ranking/position.py).
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from shopfeed.config import CATEGORIES, DAY, HOUR
from shopfeed.features.batch import AFF_COLS, BatchState, CONTENT_COLS, PROFILE_COLS
from shopfeed.features.streaming import StreamingStore

FEATURE_NAMES: List[str] = (
    AFF_COLS                                     # 16 user category affinities
    + ["u_log_purchases", "u_log_views_7d", "u_shop_pay"]
    + ["p_log_price", "p_log_age_days", "p_log_views_1h", "p_log_orders_1h",
       "p_log_purchases", "p_hide_rate", "p_is_consumable"]
    + ["m_fulfilment_days", "m_return_rate", "m_review_score", "m_followed"]
    + ["x_cat_affinity", "x_content_match"]      # affinity for this category; profile . content
    + PROFILE_COLS                               # 8 user content profile
    + CONTENT_COLS                               # 8 product content
    + ["position"]
)
POSITION_IDX = len(FEATURE_NAMES) - 1
N_FEATURES = len(FEATURE_NAMES)
DEFAULT_FILL = 0.0  # NaN policy: any missing value becomes 0.0, documented here, enforced in build_features
_CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}


def build_features(
    pairs: pd.DataFrame,          # columns: user_id, product_id, ts, position
    batch: BatchState,
    streaming: StreamingStore,
    products: pd.DataFrame,
    follows: pd.DataFrame,
) -> np.ndarray:
    n = len(pairs)
    if n == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    u = pairs[["user_id"]].merge(batch.users, on="user_id", how="left")
    p = pairs[["product_id"]].merge(products, on="product_id", how="left").merge(batch.products, on="product_id", how="left")
    m = p[["merchant_id"]].merge(batch.merchants, on="merchant_id", how="left")
    ts = pairs.ts.values.astype(float)

    aff = u[AFF_COLS].values
    aff = np.where(np.isnan(aff), 1.0 / len(CATEGORIES), aff)
    cat_idx = p.category.map(_CAT_INDEX).fillna(0).astype(int).values
    x_cat_aff = aff[np.arange(n), cat_idx]

    fkey = set(zip(follows.user_id, follows.merchant_id)) if len(follows) else set()
    followed = np.array([(uu, mm) in fkey for uu, mm in zip(pairs.user_id.values, p.merchant_id.values)], dtype=float)

    prof = np.nan_to_num(u[PROFILE_COLS].values.astype(float))
    cont = np.nan_to_num(p[CONTENT_COLS].values.astype(float))
    x_content = np.einsum("ij,ij->i", prof, cont) / np.sqrt(prof.shape[1])

    views_1h = streaming.product_count("view", pairs.product_id.values, ts, HOUR)
    orders_1h = streaming.product_count("purchase", pairs.product_id.values, ts, HOUR)
    age_days = np.clip((ts - p.published_at.values.astype(float)) / DAY, 0, None)

    cols = [
        aff,
        np.log1p(u.n_purchases.values.astype(float))[:, None],
        np.log1p(u.n_views_7d.values.astype(float))[:, None],
        u.shop_pay_history.values.astype(float)[:, None],
        np.log(p.price.values.astype(float))[:, None],
        np.log1p(age_days)[:, None],
        np.log1p(views_1h)[:, None],
        np.log1p(orders_1h)[:, None],
        np.log1p(p.n_purchases.values.astype(float))[:, None],
        p.hide_rate.values.astype(float)[:, None],
        p.is_consumable.values.astype(float)[:, None],
        m.fulfilment_days.values.astype(float)[:, None],
        m.return_rate.values.astype(float)[:, None],
        m.review_score.values.astype(float)[:, None],
        followed[:, None],
        x_cat_aff[:, None],
        x_content[:, None],
        u[PROFILE_COLS].values.astype(float),
        p[CONTENT_COLS].values.astype(float),
        pairs.position.values.astype(float)[:, None],
    ]
    X = np.concatenate(cols, axis=1).astype(np.float32)
    assert X.shape[1] == N_FEATURES, (X.shape, N_FEATURES)
    return np.nan_to_num(X, nan=DEFAULT_FILL, posinf=DEFAULT_FILL, neginf=DEFAULT_FILL)
