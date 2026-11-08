"""Candidate sources + eligibility. Eligibility is applied BEFORE any ranking.

Every source returns a DataFrame(product_id, source, score) with score in [0, 1] so
sources are comparable after the merge. The merge dedupes keeping the max score and the
list of sources an item came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from shopfeed.config import COMPLEMENTS, DAY, EligibilityConfig, SourceCaps
from shopfeed.features.streaming import StreamingStore
from shopfeed.retrieval.ann import BruteForceIndex

EMPTY = pd.DataFrame({"product_id": pd.Series(dtype=object), "source": pd.Series(dtype=object), "score": pd.Series(dtype=float)})


@dataclass
class UserContext:
    user_id: str
    country: str
    now: float
    blocked_merchants: Set[str] = field(default_factory=set)
    followed_merchants: Set[str] = field(default_factory=set)
    recent_purchased: Set[str] = field(default_factory=set)   # product ids purchased in repeat_block window
    recent_shown: Set[str] = field(default_factory=set)       # product ids shown in repeat_block window
    hidden: Set[str] = field(default_factory=set)             # product ids ever hidden -> never again
    purchase_last_ts: Dict[str, float] = field(default_factory=dict)  # product_id -> last purchase ts (all time)
    purchased_categories: Set[str] = field(default_factory=set)
    user_vec: Optional[np.ndarray] = None
    is_cold: bool = False


def enrich_products(products: pd.DataFrame, merchants: pd.DataFrame) -> pd.DataFrame:
    """Join merchant status once so eligibility is a pure column operation."""
    m = merchants[["merchant_id", "status"]].rename(columns={"status": "merchant_status"})
    return products.merge(m, on="merchant_id", how="left")


def eligibility_mask(products: pd.DataFrame, ctx: UserContext, cfg: EligibilityConfig) -> np.ndarray:
    ships = np.array([ctx.country in s for s in products.ships_to.values], dtype=bool)
    age = ctx.now - products.published_at.values.astype(float)
    mask = (
        products.in_stock.values.astype(bool)
        & ships
        & (products.merchant_status.values == "active")
        & ~products.merchant_id.isin(ctx.blocked_merchants).values
        & ~products.product_id.isin(ctx.recent_purchased).values
        & ~products.product_id.isin(ctx.recent_shown).values
        & ~products.product_id.isin(ctx.hidden).values
        & (age >= 0)
        & ((age <= cfg.max_item_age_s) | products.is_evergreen.values.astype(bool))
    )
    return mask


def _frame(pids, source: str, scores) -> pd.DataFrame:
    if len(pids) == 0:
        return EMPTY.copy()
    return pd.DataFrame({"product_id": np.asarray(pids, dtype=object), "source": source, "score": np.asarray(scores, dtype=float)})


def source_two_tower(ctx: UserContext, index: BruteForceIndex, mask: np.ndarray, cap: int) -> pd.DataFrame:
    if ctx.user_vec is None or len(index) == 0:
        return EMPTY.copy()
    ids, scores = index.search(ctx.user_vec, cap, mask=mask)
    return _frame(ids, "two_tower", (scores + 1.0) / 2.0)


def source_followed(ctx: UserContext, products: pd.DataFrame, mask: np.ndarray, cap: int) -> pd.DataFrame:
    if not ctx.followed_merchants:
        return EMPTY.copy()
    sel = products[mask & products.merchant_id.isin(ctx.followed_merchants).values]
    sel = sel.sort_values("published_at", ascending=False).head(cap)
    recency = np.exp(-(ctx.now - sel.published_at.values) / (14 * DAY))
    return _frame(sel.product_id.values, "followed", recency)


def source_complements(ctx: UserContext, products: pd.DataFrame, mask: np.ndarray, popularity: pd.Series, cap: int) -> pd.DataFrame:
    wanted: Set[str] = set()
    for c in ctx.purchased_categories:
        wanted.update(COMPLEMENTS.get(c, []))
    wanted -= ctx.purchased_categories  # complements, never the same category as the purchase
    if not wanted:
        return EMPTY.copy()
    sel = products[mask & products.category.isin(wanted).values]
    if len(sel) == 0:
        return EMPTY.copy()
    pop = popularity.reindex(sel.product_id.values).fillna(0).values
    order = np.argsort(-pop)[:cap]
    sel = sel.iloc[order]
    pop = pop[order]
    return _frame(sel.product_id.values, "complements", np.log1p(pop) / (np.log1p(pop).max() or 1.0))


def source_replenish(ctx: UserContext, products: pd.DataFrame, mask_no_repeat: np.ndarray, cap: int, lo: float = 0.8, hi: float = 1.3) -> pd.DataFrame:
    """Consumables the user bought whose repurchase window is open: now - last_ts in [lo, hi] * cycle.

    Uses a mask that does NOT exclude previously purchased items (that exclusion is what this
    source is for). Still excludes out-of-stock, shipping, suspended, blocked and hidden.
    """
    if not ctx.purchase_last_ts:
        return EMPTY.copy()
    sel = products[mask_no_repeat & products.is_consumable.values.astype(bool) & products.product_id.isin(ctx.purchase_last_ts).values]
    if len(sel) == 0:
        return EMPTY.copy()
    since = ctx.now - np.array([ctx.purchase_last_ts[p] for p in sel.product_id.values])
    cycle = sel.repurchase_cycle_s.values.astype(float)
    ratio = since / cycle
    ok = (ratio >= lo) & (ratio <= hi)
    sel = sel[ok]
    score = 1.0 - np.abs(ratio[ok] - 1.0)  # closest to the cycle end scores highest
    order = np.argsort(-score)[:cap]
    return _frame(sel.product_id.values[order], "replenish", score[order])


def source_fresh(ctx: UserContext, products: pd.DataFrame, mask: np.ndarray, cap: int, window_s: float = DAY) -> pd.DataFrame:
    age = ctx.now - products.published_at.values.astype(float)
    sel = products[mask & (age <= window_s) & (age >= 0)]
    sel = sel.sort_values("published_at", ascending=False).head(cap)
    return _frame(sel.product_id.values, "fresh", 1.0 - (ctx.now - sel.published_at.values) / window_s)


def source_trending(ctx: UserContext, products: pd.DataFrame, mask: np.ndarray, streaming: StreamingStore, cap: int) -> pd.DataFrame:
    sel = products[mask]
    if len(sel) == 0:
        return EMPTY.copy()
    orders = streaming.product_count_now("purchase", sel.product_id.values, ctx.now, DAY)
    views = streaming.product_count_now("view", sel.product_id.values, ctx.now, DAY)
    velocity = 3.0 * orders + views
    if velocity.max() <= 0:
        return EMPTY.copy()
    order = np.argsort(-velocity)[:cap]
    order = order[velocity[order] > 0]
    return _frame(sel.product_id.values[order], "trending", velocity[order] / velocity.max())


def merge_candidates(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame({"product_id": pd.Series(dtype=object), "score": pd.Series(dtype=float), "source": pd.Series(dtype=object), "sources": pd.Series(dtype=object)})
    allc = pd.concat(frames, ignore_index=True)
    allc = allc.sort_values("score", ascending=False, kind="stable")
    best = allc.drop_duplicates("product_id", keep="first")[["product_id", "score", "source"]]
    srcs = allc.groupby("product_id").source.apply(lambda s: sorted(set(s))).rename("sources")
    return best.merge(srcs, on="product_id", how="left").reset_index(drop=True)


def generate_candidates(
    ctx: UserContext,
    products: pd.DataFrame,          # enriched (merchant_status column)
    index: BruteForceIndex,
    streaming: StreamingStore,
    popularity: pd.Series,           # product_id -> n_purchases
    caps: SourceCaps,
    elig: EligibilityConfig,
    fresh_window_s: float = DAY,
) -> pd.DataFrame:
    mask = eligibility_mask(products, ctx, elig)
    # Replenishables: the user already bought it, so neither the repeat block nor the catalog age cap applies.
    ctx_no_repeat = UserContext(**{**ctx.__dict__, "recent_purchased": set(), "recent_shown": set()})
    mask_no_repeat = eligibility_mask(products, ctx_no_repeat, EligibilityConfig(repeat_block_s=elig.repeat_block_s, max_item_age_s=float("inf")))
    frames = [
        source_two_tower(ctx, index, mask, caps.two_tower),
        source_followed(ctx, products, mask, caps.followed),
        source_complements(ctx, products, mask, popularity, caps.complements),
        source_replenish(ctx, products, mask_no_repeat, caps.replenish),
        source_fresh(ctx, products, mask, caps.fresh, fresh_window_s),
        source_trending(ctx, products, mask, streaming, caps.trending),
    ]
    return merge_candidates(frames)
