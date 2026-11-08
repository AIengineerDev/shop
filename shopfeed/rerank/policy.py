"""Re-ranking policy. Pure function: candidates with values in, page with reasons out.

Order of enforcement:
  1. value order (greedy)
  2. diversity: at most `max_consecutive_merchant` in a row; at most `max_same_category_top10` in the top 10
  3. freshness floor: at least `freshness_floor` items younger than `fresh_window_s` if any exist
  4. exploration: `exploration_slots` items drawn uniformly from the eligible pool not already on the
     page, logged with propensity = k / |pool| so they can be IPS-weighted later
  5. merchant concentration cap: no merchant gets more than `merchant_share_cap` of the page
Consecutive-merchant and merchant-cap rules are hard. The category rule is relaxed only when the
page would otherwise be short. Eligibility is NOT re-checked here: candidates are already eligible.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd

from shopfeed.config import RerankConfig

REQUIRED = ["product_id", "merchant_id", "category", "published_at", "value", "source"]
OUT_COLS = REQUIRED + ["is_exploration", "propensity", "reason", "position"]


def _empty_page() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in OUT_COLS})


def _merchant_cap(cfg: RerankConfig) -> int:
    return max(1, math.floor(cfg.merchant_share_cap * cfg.page_size))


def _greedy(cands: pd.DataFrame, n: int, cfg: RerankConfig, relax_category: bool) -> List[int]:
    chosen: List[int] = []
    merchant_total: dict = {}
    category_top: dict = {}
    for i in range(len(cands)):
        if len(chosen) >= n:
            break
        row = cands.iloc[i]
        m, c = row.merchant_id, row.category
        if merchant_total.get(m, 0) >= _merchant_cap(cfg):
            continue
        last = [cands.iloc[j].merchant_id for j in chosen[-cfg.max_consecutive_merchant:]]
        if len(last) == cfg.max_consecutive_merchant and all(x == m for x in last):
            continue
        if not relax_category and len(chosen) < 10 and category_top.get(c, 0) >= cfg.max_same_category_top10:
            continue
        chosen.append(i)
        merchant_total[m] = merchant_total.get(m, 0) + 1
        if len(chosen) <= 10:
            category_top[c] = category_top.get(c, 0) + 1
    return chosen


def _take_under_cap(df: pd.DataFrame, counts: dict, cap: int, k: int) -> pd.DataFrame:
    """First k rows of df whose merchant stays under cap, updating counts as rows are taken."""
    rows = []
    for i in range(len(df)):
        if len(rows) >= k:
            break
        m = df.iloc[i].merchant_id
        if counts.get(m, 0) < cap:
            rows.append(i)
            counts[m] = counts.get(m, 0) + 1
    return df.iloc[rows]


def enforce_consecutive(page: pd.DataFrame, max_run: int) -> pd.DataFrame:
    """Scan; when an item would extend a merchant run past max_run, swap in the next item with a
    different merchant. If none exists, drop the offender. Keeps the page a subset of its input."""
    rows = [r for _, r in page.iterrows()]
    out: List[pd.Series] = []
    i = 0
    while i < len(rows):
        r = rows[i]
        run = out[-max_run:]
        if len(run) == max_run and all(x.merchant_id == r.merchant_id for x in run):
            j = next((k for k in range(i + 1, len(rows)) if rows[k].merchant_id != r.merchant_id), None)
            if j is None:
                i += 1
                continue
            rows[i], rows[j] = rows[j], rows[i]
            continue
        out.append(r)
        i += 1
    return pd.DataFrame(out).reset_index(drop=True) if out else page.iloc[0:0]


def rerank(
    candidates: pd.DataFrame,
    cfg: RerankConfig,
    now: float,
    rng: Optional[np.random.Generator] = None,
    exploration_pool: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """`candidates` must have REQUIRED columns. `exploration_pool` defaults to the candidates
    themselves (anything eligible but not chosen)."""
    missing = [c for c in REQUIRED if c not in candidates.columns]
    if missing:
        raise ValueError(f"candidates missing columns {missing}")
    if len(candidates) == 0:
        return _empty_page()
    rng = rng or np.random.default_rng(0)
    cands = candidates.drop_duplicates("product_id").sort_values("value", ascending=False, kind="stable").reset_index(drop=True)
    pool_df = exploration_pool if exploration_pool is not None else cands
    k_explore = min(cfg.exploration_slots, max(0, len(pool_df) - (cfg.page_size - cfg.exploration_slots)))
    n_main = cfg.page_size - k_explore

    chosen = _greedy(cands, n_main, cfg, relax_category=False)
    reasons = {i: "value" for i in chosen}
    if len(chosen) < n_main:
        rest = cands.drop(index=chosen).reset_index()
        extra = _greedy(rest, n_main - len(chosen), cfg, relax_category=True)
        for e in extra:
            orig = int(rest.iloc[e]["index"])
            chosen.append(orig)
            reasons[orig] = "value_relaxed_category"
    page = cands.iloc[chosen].copy()
    page["reason"] = [reasons[i] for i in chosen]

    # --- freshness floor ---
    is_fresh_all = (now - cands.published_at.astype(float)) <= cfg.fresh_window_s
    fresh_in_page = int(((now - page.published_at.astype(float)) <= cfg.fresh_window_s).sum())
    need = cfg.freshness_floor - fresh_in_page
    if need > 0:
        fresh_avail = cands[is_fresh_all & ~cands.product_id.isin(page.product_id)]
        if len(fresh_avail):
            non_fresh = page[(now - page.published_at.astype(float)) > cfg.fresh_window_s].sort_values("value")
            can_drop = len(non_fresh) if len(page) >= n_main else 0
            # Drop k lowest-value items and take k fresh ones under the merchant cap; k shrinks until
            # both sides agree so the page never ends up short because of the floor.
            for k in range(min(need, len(fresh_avail), max(can_drop, 0) if len(page) >= n_main else need), 0, -1):
                drop_ids = non_fresh.product_id.head(k) if len(page) >= n_main else pd.Series([], dtype=object)
                kept = page[~page.product_id.isin(drop_ids)]
                counts = kept.merchant_id.value_counts().to_dict()
                take = _take_under_cap(fresh_avail, counts, _merchant_cap(cfg), k).copy()
                if len(take) == k:
                    take["reason"] = "freshness_floor"
                    page = pd.concat([kept, take], ignore_index=True).sort_values("value", ascending=False, kind="stable")
                    break
    page["is_exploration"] = False
    page["propensity"] = 1.0

    # --- exploration slots ---
    if k_explore > 0:
        counts = page.merchant_id.value_counts().to_dict()
        pool = pool_df[~pool_df.product_id.isin(page.product_id)]
        pool = pool[pool.merchant_id.map(lambda m: counts.get(m, 0) < _merchant_cap(cfg))]
        if len(pool):
            shuffled = pool.iloc[rng.permutation(len(pool))]
            pick = _take_under_cap(shuffled, counts, _merchant_cap(cfg), k_explore).copy()
            k = len(pick)
            pick["reason"] = "exploration"
            pick["is_exploration"] = True
            pick["propensity"] = k / len(pool)
            for c in REQUIRED:
                if c not in pick.columns:
                    pick[c] = np.nan
            page = pd.concat([page, pick[page.columns]], ignore_index=True)
            # spread exploration items into the page rather than stacking them at the bottom
            slots = sorted(rng.choice(len(page), size=k, replace=False))
            order = [i for i in range(len(page)) if not page.iloc[i].is_exploration]
            explore_idx = [i for i in range(len(page)) if page.iloc[i].is_exploration]
            final: List[int] = []
            e = 0
            for pos in range(len(page)):
                if e < k and pos == slots[e]:
                    final.append(explore_idx[e]); e += 1
                elif order:
                    final.append(order.pop(0))
            final.extend(order)
            page = page.iloc[final].reset_index(drop=True)

    page = enforce_consecutive(page, cfg.max_consecutive_merchant)
    page = page.head(cfg.page_size).reset_index(drop=True)
    page["position"] = np.arange(len(page))
    return page[OUT_COLS]
