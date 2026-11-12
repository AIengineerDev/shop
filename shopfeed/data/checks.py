"""Spec §9 data-readiness checks. Each check returns a CheckResult; `run_all` runs them all.

Profiles (config.GATES): "production" uses the spec thresholds; "sample" is for the
45-day feed sample (dev plan §6). Columns in KNOWN_MISSING are absent from the feed log by
design (spec §2 build mapping) and are exempt from the null-rate check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from shopfeed import config, labels
from shopfeed.data.build import Sources
from shopfeed.schema import COLUMNS, GROUP_KEYS, LABELS

KNOWN_MISSING = {"bm25", "rating"}
FEATURE_COLUMNS = ["bm25", "semantic_sim", "price", "hist_ctr_7d", "rating"]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str

    def __post_init__(self):
        self.passed = bool(self.passed)  # numpy bools are not JSON-serialisable


def check_header(df):
    return CheckResult("header", list(df.columns) == COLUMNS, f"{len(df.columns)} columns")


def check_scale(df, gates):
    rows, groups = len(df), df["qid"].nunique()
    ok = rows >= gates["min_rows"] and groups >= gates["min_groups"]
    return CheckResult("scale", ok, f"rows={rows} (>= {gates['min_rows']}), "
                                    f"groups={groups} (>= {gates['min_groups']})")


def check_nulls(df):
    key_nulls = [c for c in GROUP_KEYS + LABELS if df[c].isna().any()]
    rates = {c: float(df[c].isna().mean()) for c in FEATURE_COLUMNS if c not in KNOWN_MISSING}
    high = {c: round(r, 4) for c, r in rates.items() if r >= 0.05}
    return CheckResult("nulls", not key_nulls and not high,
                       f"key/label nulls={key_nulls}, feature null rates>=5%={high}, "
                       f"exempt={sorted(KNOWN_MISSING)}")


def check_label_consistency(df):
    n = int(labels.consistency_violations(df).sum())
    return CheckResult("label_consistency", n == 0, f"violations={n}")


def check_groups(df):
    sizes = df.groupby("qid").size()
    dup_pos = int(df.duplicated(["qid", "displayed_pos"]).sum())
    gaps = int((df.groupby("qid")["displayed_pos"].max() != sizes).sum())
    ok = bool((sizes >= 2).all()) and dup_pos == 0
    return CheckResult("groups", ok, f"min_size={sizes.min()}, duplicate positions={dup_pos}, "
                                     f"groups with position gaps (documented)={gaps}")


def check_semantic_direction(df):
    r = df[["semantic_sim", "clicked"]].corr().iloc[0, 1]
    return CheckResult("semantic_sim_direction", bool(r > 0),
                       f"corr(semantic_sim, clicked)={r:.3f}")


def check_hist_ctr_point_in_time(df, src: Sources | None, n=1000):
    """Recompute hist_ctr_7d naively for a sample, from raw events, and compare."""
    if src is None:
        return CheckResult("hist_ctr_point_in_time", True, "skipped: no sources")
    sample = df.sample(min(n, len(df)), random_state=config.SEED)
    imp, ev = src.impressions, src.events
    clicks = ev[(ev["type"] == "view") & ev["impression_id"].notna()]
    mism = 0
    for _, row in sample.iterrows():
        t, item = row["impression_ts"], row["item_id"]
        lo = t - 7 * 86400
        i = imp[(imp["product_id"] == item) & (imp["ts"] >= lo) & (imp["ts"] < t)]
        c = clicks[(clicks["product_id"] == item) & (clicks["ts"] >= lo) & (clicks["ts"] < t)]
        expected = len(c) / len(i) if len(i) else np.nan
        got = row["hist_ctr_7d"]
        if not ((np.isnan(expected) and np.isnan(got)) or np.isclose(expected, got)):
            mism += 1
    return CheckResult("hist_ctr_point_in_time", mism == 0, f"mismatches={mism}/{len(sample)}")


def check_position_bias(df):
    ctr = df.groupby("displayed_pos")["clicked"].mean()
    c1, c10 = ctr.get(1, np.nan), ctr.get(10, np.nan)
    return CheckResult("position_bias", bool(c1 > 1.5 * c10), f"CTR@1={c1:.3f}, CTR@10={c10:.3f}")


def check_grade_distribution(df):
    d = df["relevance_grade"].value_counts(normalize=True)
    g0, g4 = d.get(0, 0.0), d.get(4, 0.0)
    ok = 0.85 <= g0 <= 0.90 and 0.01 <= g4 <= 0.02
    dist = {int(k): round(float(v), 4) for k, v in d.sort_index().items()}
    return CheckResult("grade_distribution", ok, f"{dist} (want g0 in [.85,.90], g4 in [.01,.02])")


def check_temporal(df):
    ok = "impression_ts" in df and df["impression_ts"].notna().all()
    return CheckResult("temporal_column", bool(ok), "impression_ts present and non-null")


def check_attribution(df, src: Sources | None, gates):
    """Share of stamped wishlist / purchase events that resolve to a row in the dataset."""
    if src is None:
        return CheckResult("attribution", True, "skipped: no sources")
    ev = src.events
    imp_ids = src.impressions[["impression_id", "request_id", "product_id"]]
    rows = df[["qid", "item_id"]].assign(
        request_id="req_" + df["qid"].astype(str), present=1)
    out = {}
    for typ in ("save", "purchase"):
        e = ev[ev["type"] == typ]
        stamped = (e.dropna(subset=["impression_id"])[["event_id", "impression_id"]]
                   .merge(imp_ids, on="impression_id"))
        hit = stamped.merge(rows, left_on=["request_id", "product_id"],
                            right_on=["request_id", "item_id"], how="left")["present"].notna()
        out[typ] = (float(hit.mean()) if len(hit) else 1.0, len(e) - len(stamped))
    ok = all(rate >= gates["min_purchase_attribution"] for rate, _ in out.values())
    detail = ", ".join(f"{k}: resolved={r:.3f}, non-search (no impression)={n}"
                       for k, (r, n) in out.items())
    return CheckResult("attribution", ok, detail)


def check_joins(df, src: Sources | None):
    if src is None:
        return CheckResult("joins", True, "skipped: no sources")
    p = src.products
    orphan_items = int((~df["item_id"].isin(p["product_id"])).sum())
    merchants = set(p["merchant_id"])
    orphan_sellers = int((~df["seller_id"].isin(merchants)).sum())
    orphan_cat = int(df["category"].isna().sum())
    ok = orphan_items == orphan_sellers == orphan_cat == 0
    return CheckResult("joins", ok, f"orphan items={orphan_items}, sellers={orphan_sellers}, "
                                    f"categories={orphan_cat}")


def run_all(df: pd.DataFrame, src: Sources | None = None,
            profile: str = "production") -> list[CheckResult]:
    gates = config.GATES[profile]
    return [
        check_header(df), check_scale(df, gates), check_nulls(df),
        check_label_consistency(df), check_groups(df), check_semantic_direction(df),
        check_hist_ctr_point_in_time(df, src), check_position_bias(df),
        check_grade_distribution(df), check_temporal(df),
        check_attribution(df, src, gates), check_joins(df, src),
    ]


def as_dicts(results):
    return [asdict(r) for r in results]
