"""Daily batch features. Computed from events strictly before `as_of` (point-in-time correct).

All functions take an explicit `as_of` timestamp; nothing here reads the wall clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from shopfeed.config import CATEGORIES, DAY
from shopfeed.data.synth import CONTENT_DIM

CONTENT_COLS: List[str] = [f"c{d}" for d in range(CONTENT_DIM)]
AFF_COLS: List[str] = [f"aff_{c}" for c in CATEGORIES]
PROFILE_COLS: List[str] = [f"prof_{d}" for d in range(CONTENT_DIM)]
# Meaningful engagements and their weight in the user profile. Views alone are too noisy.
ENGAGEMENT_WEIGHTS = {"dwell": 1.0, "save": 2.0, "add_to_cart": 2.0, "purchase": 3.0}


@dataclass
class BatchState:
    as_of: float
    users: pd.DataFrame      # user_id, aff_*, prof_*, n_purchases, n_views_7d, shop_pay_history
    products: pd.DataFrame   # product_id, n_purchases, n_impr, hide_rate
    merchants: pd.DataFrame  # merchant_id, fulfilment_days, return_rate, review_score, is_suspended


def compute_batch_state(
    events: pd.DataFrame,
    impressions: pd.DataFrame,
    users: pd.DataFrame,
    products: pd.DataFrame,
    merchants: pd.DataFrame,
    as_of: float,
) -> BatchState:
    ev = events[events.ts < as_of]
    imp = impressions[impressions.ts < as_of]
    purchases = ev[ev.type == "purchase"].merge(products[["product_id", "category"] + CONTENT_COLS], on="product_id", how="inner")
    engaged = ev[ev.type.isin(ENGAGEMENT_WEIGHTS) & ev.product_id.notna()].merge(
        products[["product_id", "category"] + CONTENT_COLS], on="product_id", how="inner")
    engaged = engaged.assign(w=engaged.type.map(ENGAGEMENT_WEIGHTS).astype(float))

    # --- user category affinity: engagement-weighted category shares, uniform prior when no engagement ---
    uniform = 1.0 / len(CATEGORIES)
    aff = pd.DataFrame(uniform, index=users.user_id, columns=AFF_COLS)
    if len(engaged):
        counts = engaged.pivot_table(index="user_id", columns="category", values="w", aggfunc="sum", fill_value=0.0)
        counts = counts.reindex(columns=CATEGORIES, fill_value=0.0)
        norm = counts.div(counts.sum(axis=1), axis=0)
        norm.columns = AFF_COLS
        aff.loc[norm.index, AFF_COLS] = norm.values

    # --- user content profile: engagement-weighted mean content vector, zero when none ---
    prof = pd.DataFrame(0.0, index=users.user_id, columns=PROFILE_COLS)
    if len(engaged):
        wc = engaged[CONTENT_COLS].mul(engaged.w, axis=0)
        wc["user_id"] = engaged.user_id.values
        mean_c = wc.groupby("user_id").sum().div(engaged.groupby("user_id").w.sum(), axis=0)
        mean_c.columns = PROFILE_COLS
        prof.loc[mean_c.index, PROFILE_COLS] = mean_c.values

    n_purch = purchases.groupby("user_id").size().reindex(users.user_id, fill_value=0)
    recent_views = ev[(ev.type == "view") & (ev.ts >= as_of - 7 * DAY)].groupby("user_id").size()
    n_views_7d = recent_views.reindex(users.user_id, fill_value=0)

    u = pd.DataFrame({"user_id": users.user_id.values})
    u[AFF_COLS] = aff.values
    u[PROFILE_COLS] = prof.values
    u["n_purchases"] = n_purch.values
    u["n_views_7d"] = n_views_7d.values
    u["shop_pay_history"] = users.shop_pay_history.astype(float).values

    # --- product popularity & hide rate from history before as_of ---
    p_purch = ev[ev.type == "purchase"].groupby("product_id").size()
    p_hide = ev[ev.type.isin(["hide", "not_interested"])].groupby("product_id").size()
    p_impr = imp.groupby("product_id").size()
    p = pd.DataFrame({"product_id": products.product_id.values})
    p["n_purchases"] = p_purch.reindex(p.product_id, fill_value=0).values
    p["n_impr"] = p_impr.reindex(p.product_id, fill_value=0).values
    hides = p_hide.reindex(p.product_id, fill_value=0).values
    p["hide_rate"] = np.where(p.n_impr > 0, hides / np.maximum(p.n_impr, 1), 0.0)

    m = merchants[["merchant_id", "fulfilment_days", "return_rate", "review_score"]].copy()
    m["is_suspended"] = (merchants.status == "suspended").astype(float).values
    return BatchState(as_of=as_of, users=u, products=p, merchants=m)
