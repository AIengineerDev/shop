"""Post-ranking business rules (spec §5.5). Each behind a flag.

Placement is a constrained greedy pass: for each position from the top, take the highest
scored remaining item allowed at that position; if none is allowed, take the highest
remaining item (rules never drop items). Then two slot rules run:

- relevance floor (on): positions 1-3 need semantic_sim >= the group median.
- unpopular demotion (on): is_unpopular items cannot occupy positions 1-5. New items
  (age < NEW_ITEM_DAYS and < UNPOPULAR_MIN_IMPRESSIONS impressions) are exempt.
- new-item exploration (on): if no new item sits in positions 6-10, the best-scored new item
  below position 10 is inserted at slot 6 + (qid % 5) (round-robin across requests).
- best-seller boost (off): a category #1 seller that the model put below position 10 is
  moved to position 10.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from shopfeed import config


@dataclass
class RuleFlags:
    relevance_floor: bool = True
    unpopular_demotion: bool = True
    new_item_exploration: bool = True
    bestseller_boost: bool = False


def _is_new(df: pd.DataFrame) -> np.ndarray:
    return ((df["item_age_days"].to_numpy() < config.NEW_ITEM_DAYS)
            & (df["item_impressions_30d"].to_numpy() < config.UNPOPULAR_MIN_IMPRESSIONS))


def _move(order: list[int], idx: int, to_pos: int) -> list[int]:
    order = [i for i in order if i != idx]
    order.insert(to_pos, idx)
    return order


def apply_rules(df: pd.DataFrame, flags: RuleFlags) -> pd.DataFrame:
    """df: one request's candidates with a `score` column plus rule inputs. Returns df
    reordered (index reset) with a 1-based `rank` column."""
    df = df.reset_index(drop=True)
    order = list(np.argsort(-df["score"].to_numpy(), kind="stable"))
    new = _is_new(df)
    sim = df["semantic_sim"].to_numpy(dtype=float)
    floor_ok = ~(sim < np.nanmedian(sim)) if np.isfinite(sim).any() else np.ones(len(df), bool)
    unpopular = (df["is_unpopular"].to_numpy() == 1) & ~new

    def allowed(i: int, pos: int) -> bool:  # pos is 1-based
        if flags.relevance_floor and pos <= 3 and not floor_ok[i]:
            return False
        return not (flags.unpopular_demotion and pos <= 5 and unpopular[i])

    if flags.relevance_floor or flags.unpopular_demotion:
        remaining, placed = order[:], []
        for pos in range(1, len(order) + 1):
            pick = next((i for i in remaining if allowed(i, pos)), remaining[0])
            placed.append(pick)
            remaining.remove(pick)
        order = placed

    no_new_in_6_10 = not any(new[i] for i in order[5:10])
    if flags.new_item_exploration and len(order) > 10 and no_new_in_6_10:
        below = [i for i in order[10:] if new[i]]
        if below:
            slot = 5 + int(df["qid"].iloc[0]) % 5  # 0-based index of positions 6-10
            order = _move(order, below[0], slot)

    if flags.bestseller_boost and len(order) > 10:
        rank_cat = df["item_bestseller_rank_category"].to_numpy()
        top = df["is_category_top20"].to_numpy() == 1
        for i in list(order[10:]):
            if rank_cat[i] == 1 and top[i]:
                order = _move(order, i, 9)
                break

    out = df.iloc[order].reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out
