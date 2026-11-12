"""Offline evaluation harness (spec §8.2-8.4).

- `bootstrap_ci`: resample query groups (qid) with replacement, 1,000 times by default.
- `paired_delta`: per-query deltas vs a reference scorer, bootstrap CI on the mean delta
  and on the relative delta, plus a Wilcoxon signed-rank p-value.
- `evaluate`: every scorer x every metric, overall and per slice, as a tidy frame.
- `write_html`: self-contained HTML report under reports/.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from shopfeed import config
from shopfeed.eval.metrics import RATIO_METRICS, add_longtail_flag, per_query, summarise

N_BOOT = 1000


def _metric_from_table(pq: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray | None]:
    """(numerator per query, denominator per query or None for plain means)."""
    if metric == "purchase_mrr":
        return pq["purchase_rr"].to_numpy(dtype=float), None
    if metric in RATIO_METRICS:
        num, den = RATIO_METRICS[metric]
        return pq[num].to_numpy(dtype=float), pq[den].to_numpy(dtype=float)
    return pq[metric].to_numpy(dtype=float), None


def _stat(num: np.ndarray, den: np.ndarray | None) -> float:
    if den is None:
        return float(np.nanmean(num)) if np.isfinite(num).any() else float("nan")
    d = den.sum()
    return float(num.sum() / d) if d else float("nan")


def bootstrap_ci(pq: pd.DataFrame, metric: str, n: int = N_BOOT, seed: int = config.SEED,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    num, den = _metric_from_table(pq, metric)
    point = _stat(num, den)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n, len(num)))
    with np.errstate(invalid="ignore"):
        if den is None:
            boots = np.nanmean(np.where(np.isfinite(num[idx]), num[idx], np.nan), axis=1)
        else:
            boots = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = np.nanquantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def paired_delta(pq_model: pd.DataFrame, pq_ref: pd.DataFrame, metric: str,
                 n: int = N_BOOT, seed: int = config.SEED) -> dict:
    """Bootstrap CI of (model - ref) over the same resampled queries, plus Wilcoxon p."""
    a_num, a_den = _metric_from_table(pq_model, metric)
    b_num, b_den = _metric_from_table(pq_ref.loc[pq_model.index], metric)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a_num), size=(n, len(a_num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        if a_den is None:
            ok = np.isfinite(a_num) & np.isfinite(b_num)
            d = np.where(ok, a_num - b_num, np.nan)
            boots = np.nanmean(d[idx], axis=1)
            ref_boot = np.nanmean(np.where(ok, b_num, np.nan)[idx], axis=1)
            point, ref_point = float(np.nanmean(d)), float(np.nanmean(b_num[ok]))
            p = float(stats.wilcoxon(d[ok]).pvalue) if (d[ok] != 0).any() else 1.0
        else:
            a = a_num[idx].sum(axis=1) / a_den[idx].sum(axis=1)
            b = b_num[idx].sum(axis=1) / b_den[idx].sum(axis=1)
            boots, ref_boot = a - b, b
            point = _stat(a_num, a_den) - _stat(b_num, b_den)
            ref_point = _stat(b_num, b_den)
            p = float("nan")
        rel = boots / ref_boot
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    rlo, rhi = np.nanquantile(rel, [0.025, 0.975])
    return {"metric": metric, "delta": point, "delta_lo": float(lo), "delta_hi": float(hi),
            "rel_delta": point / ref_point if ref_point else float("nan"),
            "rel_lo": float(rlo), "rel_hi": float(rhi), "wilcoxon_p": p}


# ---- slices ------------------------------------------------------------------------------

def query_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Per-qid slice keys (spec §8.2)."""
    first = df.groupby("qid", sort=False).first()
    attrs = pd.DataFrame(index=first.index)
    for col in ("device", "user_aov_tier"):
        if col in first:
            attrs[col] = first[col]
    if "is_new_user" in first:
        attrs["user_type"] = np.where(first["is_new_user"] == 1, "new", "returning")
    size = df.groupby("qid", sort=False).size()
    attrs["candidate_set_size"] = pd.cut(size, [0, 10, 19, 1000], labels=["<=10", "11-19", "20+"])
    if "query_text" in first:
        terms = first["query_text"].fillna("").str.split().str.len()
        attrs["query_length"] = pd.cut(terms, [-1, 0, 1, 3, 1000],
                                       labels=["empty", "1", "2-3", "4+"])
    bought = df[df["purchased"] == 1]
    if "is_category_top20" in df and len(bought):
        pop = bought.groupby("qid")["is_category_top20"].max().map({1: "bestseller", 0: "longtail"})
        attrs["purchased_item_popularity"] = pop.reindex(attrs.index).fillna("no_purchase")
    if "category" in df and len(bought):
        attrs["purchased_category"] = bought.groupby("qid")["category"].first().reindex(
            attrs.index).fillna("no_purchase")
    if "seller_item_count" in df:
        sz = df.groupby("qid")["seller_item_count"].median()
        attrs["seller_size"] = pd.qcut(sz.rank(method="first"), 3, labels=["small", "mid", "large"])
    return attrs.astype("string")


# ---- full evaluation -------------------------------------------------------------------

def evaluate(df: pd.DataFrame, scorers: list[str], reference: str | None = None,
             metrics: list[str] | None = None, n_boot: int = N_BOOT,
             slices: bool = True) -> dict[str, pd.DataFrame]:
    """Returns {"overall": ..., "deltas": ..., "slices": ...} tidy frames."""
    df = add_longtail_flag(df)
    tables = {s: per_query(df, s) for s in scorers}
    metrics = metrics or [m for m in summarise(next(iter(tables.values())))
                          if not np.isnan(summarise(next(iter(tables.values())))[m])]
    overall = []
    for s, pq in tables.items():
        for m in metrics:
            point, lo, hi = bootstrap_ci(pq, m, n_boot)
            overall.append({"scorer": s, "metric": m, "value": point, "ci_lo": lo, "ci_hi": hi})
    result = {"overall": pd.DataFrame(overall)}

    if reference:
        deltas = [dict(scorer=s, **paired_delta(pq, tables[reference], m, n_boot))
                  for s, pq in tables.items() if s != reference for m in metrics]
        result["deltas"] = pd.DataFrame(deltas)

    if slices:
        attrs = query_attributes(df)
        rows = []
        for col in attrs.columns:
            for val, qids in attrs.groupby(col, observed=True).groups.items():
                for s, pq in tables.items():
                    summary = summarise(pq.loc[pq.index.intersection(qids)])
                    for m in metrics:
                        rows.append({"slice": col, "value_of_slice": val, "scorer": s,
                                     "metric": m, "value": summary.get(m, np.nan),
                                     "n_queries": len(qids)})
        result["slices"] = pd.DataFrame(rows)
    return result


def write_html(result: dict[str, pd.DataFrame], path: Path, title: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title>",
             "<style>body{font:14px system-ui;margin:24px;max-width:1200px}"
             "table{border-collapse:collapse;margin:12px 0}td,th{padding:4px 8px;"
             "border-bottom:1px solid #ddd;text-align:right}th{background:#f4f4f4}</style>",
             f"<h1>{html.escape(title)}</h1>"]
    o = result["overall"].pivot(index="metric", columns="scorer", values="value")
    parts += ["<h2>Overall (bootstrap 95% CI in raw table below)</h2>",
              o.to_html(float_format="%.4f"), result["overall"].to_html(float_format="%.4f",
                                                                        index=False)]
    if "deltas" in result:
        parts += ["<h2>Paired deltas vs reference</h2>",
                  result["deltas"].to_html(float_format="%.4f", index=False)]
    if "slices" in result:
        sl = result["slices"]
        for m in ("ndcg@10", "purchase_mrr"):
            sub = sl[sl["metric"] == m]
            if len(sub):
                t = sub.pivot_table(index=["slice", "value_of_slice", "n_queries"],
                                    columns="scorer", values="value", observed=True)
                parts += [f"<h2>Slices: {m}</h2>", t.to_html(float_format="%.4f")]
    path.write_text("\n".join(parts))
    return path
