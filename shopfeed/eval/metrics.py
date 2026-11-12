"""Ranking metrics (spec §8.1). Per-query functions take arrays for ONE query group.

Conventions:
- Items are ordered by descending score; ties keep input order (stable sort). Callers that
  need a deterministic tie-break sort rows by item_id first.
- NDCG uses gain[grade] / log2(rank + 1). Queries with zero ideal DCG return NaN and are
  excluded from means.
- Purchase MRR is 1 / rank of the best-ranked purchased item; NaN when nothing was bought.
- Purchase@k is micro-averaged across queries: purchased items in top k / purchased items.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shopfeed import config


def _order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores, dtype=float), kind="stable")


def dcg_at_k(grades_in_rank_order: np.ndarray, k: int, gain) -> float:
    g = np.asarray(gain, dtype=float)[np.asarray(grades_in_rank_order[:k], dtype=int)]
    return float((g / np.log2(np.arange(2, len(g) + 2))).sum())


def ndcg_at_k(grades, scores, k: int = 10, gain=config.GAIN_PURCHASE_DOMINANT) -> float:
    grades = np.asarray(grades, dtype=int)
    ideal = dcg_at_k(np.sort(grades)[::-1], k, gain)
    if ideal == 0:
        return float("nan")
    return dcg_at_k(grades[_order(scores)], k, gain) / ideal


def purchase_reciprocal_rank(purchased, scores) -> float:
    p = np.asarray(purchased)[_order(scores)]
    hits = np.flatnonzero(p == 1)
    return float("nan") if not len(hits) else 1.0 / (hits[0] + 1)


def purchase_hits_at_k(purchased, scores, k: int) -> tuple[int, int]:
    p = np.asarray(purchased)[_order(scores)]
    return int(p[:k].sum()), int(p.sum())


def average_precision(grades, scores, min_grade: int = 2) -> float:
    rel = (np.asarray(grades)[_order(scores)] >= min_grade).astype(float)
    if not rel.sum():
        return float("nan")
    precision = np.cumsum(rel) / np.arange(1, len(rel) + 1)
    return float((precision * rel).sum() / rel.sum())


def top_k_mask(scores, k: int) -> np.ndarray:
    m = np.zeros(len(scores), dtype=bool)
    m[_order(scores)[:k]] = True
    return m


# ---- per-query table -------------------------------------------------------------------

def per_query(df: pd.DataFrame, score_col: str, k: int = 10) -> pd.DataFrame:
    """One row per qid with every §8.1 metric for `score_col`.

    Optional columns used when present: item_purchases_30d (long-tail exposure),
    item_age_days (new-item exposure), is_category_top20 (best-seller coverage), price.
    """
    has = set(df.columns)
    out = []
    for qid, g in df.groupby("qid", sort=False):
        s, grades = g[score_col].to_numpy(), g["relevance_grade"].to_numpy()
        purchased = g["purchased"].to_numpy()
        top = top_k_mask(s, k)
        hits, n_p = purchase_hits_at_k(purchased, s, k)
        row = {
            "qid": qid,
            "ndcg@5": ndcg_at_k(grades, s, 5), f"ndcg@{k}": ndcg_at_k(grades, s, k),
            "ndcg@20": ndcg_at_k(grades, s, 20),
            f"ndcg@{k}_exp": ndcg_at_k(grades, s, k, config.GAIN_EXPONENTIAL),
            "purchase_rr": purchase_reciprocal_rank(purchased, s),
            "purchase_hits@1": purchase_hits_at_k(purchased, s, 1)[0],
            "purchase_hits@3": purchase_hits_at_k(purchased, s, 3)[0],
            f"purchase_hits@{k}": hits, "purchases": n_p,
            "map_g2": average_precision(grades, s),
        }
        if "price" in has:
            row[f"mean_price@{k}"] = float(g["price"].to_numpy()[top].mean())
        if "item_age_days" in has:
            row[f"new_item_share@{k}"] = float(
                (g["item_age_days"].to_numpy()[top] < config.NEW_ITEM_DAYS).mean())
        if "is_longtail" in has:
            row[f"longtail_share@{k}"] = float(g["is_longtail"].to_numpy()[top].mean())
        if "is_category_top20" in has:
            bs = g["is_category_top20"].to_numpy() == 1
            row[f"bestseller_covered@{k}"] = float((bs & top).any()) if bs.any() else np.nan
        out.append(row)
    return pd.DataFrame(out).set_index("qid")


RATIO_METRICS = {  # metric -> (numerator column, denominator column), micro-averaged
    "purchase@1": ("purchase_hits@1", "purchases"),
    "purchase@3": ("purchase_hits@3", "purchases"),
    "purchase@10": ("purchase_hits@10", "purchases"),
}
MEAN_METRICS = ["ndcg@5", "ndcg@10", "ndcg@20", "ndcg@10_exp", "purchase_rr", "map_g2",
                "mean_price@10", "new_item_share@10", "longtail_share@10",
                "bestseller_covered@10"]


def summarise(pq: pd.DataFrame) -> dict[str, float]:
    """Collapse a per-query table into scalar metrics (NaN-aware means, micro ratios)."""
    out = {m: float(pq[m].mean()) for m in MEAN_METRICS if m in pq}
    out["purchase_mrr"] = out.pop("purchase_rr", float("nan"))
    for name, (num, den) in RATIO_METRICS.items():
        if num in pq:
            d = pq[den].sum()
            out[name] = float(pq[num].sum() / d) if d else float("nan")
    return out


def add_longtail_flag(df: pd.DataFrame, top_share: float = 0.2) -> pd.DataFrame:
    """is_longtail = item outside the top `top_share` of items by item_purchases_30d."""
    if "item_purchases_30d" not in df:
        return df
    per_item = df.groupby("item_id")["item_purchases_30d"].max()
    cut = per_item.quantile(1 - top_share)
    return df.assign(is_longtail=(df["item_purchases_30d"] <= cut).astype(int))
