"""Parquet -> ranking_items.csv (spec §2 column mapping, dev-plan M1).

Sources: impressions, events, products, users (data/*.parquet). One output row per
(request_id, product_id). Mapping decisions recorded in spec §2 "Build mapping":

- qid = integer part of request_id; session_id from SESSION_GAP_S inactivity per user.
- displayed_pos = position + 1. Duplicate (request, product) impressions are merged:
  earliest position kept, labels = max over the duplicates (leaves position gaps).
- Events with a stamped impression_id are trusted up to STAMPED_MAX_LAG_S. Events without
  one are attributed to the user's most recent impression of the item within
  ATTRIBUTION_WINDOW_S; otherwise they are non-search events and ignored.
- clicked <- view, wishlisted <- save, added_to_cart <- add_to_cart, purchased <- purchase,
  hide demotes a click-only row to grade 0.
- semantic_sim = cosine(user t0..t7, product c0..c7): a USER-ITEM proxy, not query-item.
- bm25, rating = NaN, query_text = "", device = "unknown": not present in the feed log.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from shopfeed import config, labels
from shopfeed.data import aov
from shopfeed.schema import COLUMNS, DTYPES

EVENT_LABEL = {"view": "clicked", "save": "wishlisted", "add_to_cart": "added_to_cart",
               "purchase": "purchased", "hide": "hidden"}
LABEL_COLS = ["clicked", "wishlisted", "added_to_cart", "purchased", "hidden"]


@dataclass
class Sources:
    impressions: pd.DataFrame
    events: pd.DataFrame
    products: pd.DataFrame
    users: pd.DataFrame

    @classmethod
    def load(cls, data_dir=config.DATA_DIR) -> Sources:
        return cls(*(pd.read_parquet(data_dir / f"{n}.parquet")
                     for n in ("impressions", "events", "products", "users")))


@dataclass
class BuildReport:
    counts: dict = field(default_factory=dict)

    def add(self, key: str, value) -> None:
        self.counts[key] = value.item() if hasattr(value, "item") else value


def sessionize(impressions: pd.DataFrame, gap_s: float = config.SESSION_GAP_S) -> pd.Series:
    """Map request_id -> session_id ("s_<n>"). New session when a user's request starts more
    than gap_s after the previous request's start."""
    req = (impressions.groupby("request_id", sort=False)
           .agg(user_id=("user_id", "first"), start=("ts", "min"))
           .sort_values(["user_id", "start"]))
    new = (req["user_id"] != req["user_id"].shift()) | (req["start"].diff() > gap_s)
    return ("s_" + new.cumsum().astype(str)).rename("session_id")


def attribute_events(impressions: pd.DataFrame, events: pd.DataFrame,
                     report: BuildReport | None = None) -> pd.DataFrame:
    """Return events joined to an impression_id with columns [impression_id, type, lag_s]."""
    ev = events[events["type"].isin(EVENT_LABEL)].copy()
    imp = impressions[["impression_id", "user_id", "product_id", "ts"]]

    stamped = ev[ev["impression_id"].notna()].merge(
        imp[["impression_id", "ts"]], on="impression_id", suffixes=("", "_imp"))
    stamped["lag_s"] = stamped["ts"] - stamped["ts_imp"]
    ok = (stamped["lag_s"] >= 0) & (stamped["lag_s"] <= config.STAMPED_MAX_LAG_S)

    unstamped = ev[ev["impression_id"].isna()].drop(columns="impression_id")
    recon = unstamped.merge(imp, on=["user_id", "product_id"], suffixes=("", "_imp"))
    recon["lag_s"] = recon["ts"] - recon["ts_imp"]
    recon = recon[(recon["lag_s"] >= 0) & (recon["lag_s"] <= config.ATTRIBUTION_WINDOW_S)]
    recon = recon.sort_values("lag_s").drop_duplicates("event_id")

    if report is not None:
        report.add("events_stamped", len(stamped))
        report.add("events_stamped_rejected_lag", int((~ok).sum()))
        report.add("events_unstamped", len(unstamped))
        report.add("events_unstamped_reconstructed", len(recon))
        p = stamped[ok & (stamped["type"] == "purchase")]
        report.add("purchases_stamped_lag_gt_24h", int((p["lag_s"] > 86400).sum()))
        report.add("purchases_stamped", len(p))
    cols = ["impression_id", "type", "lag_s"]
    return pd.concat([stamped.loc[ok, cols], recon[cols]], ignore_index=True)


def impression_labels(impressions: pd.DataFrame, attributed: pd.DataFrame) -> pd.DataFrame:
    flags = (attributed.assign(col=attributed["type"].map(EVENT_LABEL), v=1)
             .pivot_table(index="impression_id", columns="col", values="v", aggfunc="max"))
    flags = flags.reindex(columns=LABEL_COLS).fillna(0).astype("int64")
    out = impressions[["impression_id"]].merge(flags, left_on="impression_id",
                                               right_index=True, how="left")
    return out.fillna({c: 0 for c in LABEL_COLS}).astype({c: "int64" for c in LABEL_COLS})


def semantic_sim(users: pd.DataFrame, products: pd.DataFrame,
                 user_ids: pd.Series, product_ids: pd.Series) -> np.ndarray:
    u = users.set_index("user_id")[[f"t{i}" for i in range(8)]].reindex(user_ids).to_numpy()
    p = products.set_index("product_id")[[f"c{i}" for i in range(8)]].reindex(product_ids)
    p = p.to_numpy()
    num = (u * p).sum(axis=1)
    den = np.linalg.norm(u, axis=1) * np.linalg.norm(p, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def hist_ctr_7d(rows: pd.DataFrame, impressions: pd.DataFrame,
                clicks: pd.DataFrame, window_s: float = 7 * 86400) -> pd.Series:
    """Item CTR over [t - window, t): clicks / impressions, both strictly before t."""
    imp_by_item = {k: np.sort(v.to_numpy()) for k, v in impressions.groupby("product_id")["ts"]}
    clk_by_item = {k: np.sort(v.to_numpy()) for k, v in clicks.groupby("product_id")["ts"]}
    out = np.full(len(rows), np.nan)
    empty = np.array([])
    for item, idx in rows.groupby("item_id").indices.items():
        t = rows["impression_ts"].to_numpy()[idx]
        it, ct = imp_by_item.get(item, empty), clk_by_item.get(item, empty)
        n_imp = np.searchsorted(it, t, "left") - np.searchsorted(it, t - window_s, "left")
        n_clk = np.searchsorted(ct, t, "left") - np.searchsorted(ct, t - window_s, "left")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[idx] = np.where(n_imp > 0, n_clk / np.maximum(n_imp, 1), np.nan)
    return pd.Series(out, index=rows.index)


def build_rows(src: Sources, report: BuildReport | None = None) -> pd.DataFrame:
    report = report or BuildReport()
    imp = src.impressions.copy()
    report.add("impressions_in", len(imp))

    attributed = attribute_events(imp, src.events, report)
    imp = imp.merge(impression_labels(imp, attributed), on="impression_id")

    # Merge duplicate (request, product) impressions.
    dup = imp.duplicated(["request_id", "product_id"], keep=False)
    report.add("duplicate_impressions_merged", int(dup.sum()))
    imp = imp.sort_values(["request_id", "position"])
    agg = {"user_id": "first", "ts": "min", "position": "min",
           **{c: "max" for c in LABEL_COLS}}
    rows = imp.groupby(["request_id", "product_id"], sort=False).agg(agg).reset_index()

    rows["clicked"] = labels.backfill_clicked(rows)
    rows["relevance_grade"] = labels.grade(rows, rows["hidden"])
    report.add("rows_hidden_demoted", int(((rows["hidden"] == 1)
                                           & (rows["clicked"] == 1)
                                           & (rows["relevance_grade"] == 0)).sum()))

    sess = sessionize(src.impressions)
    rows["session_id"] = rows["request_id"].map(sess)
    rows["qid"] = rows["request_id"].str.extract(r"(\d+)$", expand=False).astype("int64")
    rows = rows.rename(columns={"product_id": "item_id", "ts": "impression_ts"})
    rows["displayed_pos"] = rows["position"].astype("int64") + 1

    prod = src.products.set_index("product_id")
    rows["price"] = rows["item_id"].map(prod["price"])
    rows["seller_id"] = rows["item_id"].map(prod["merchant_id"])
    rows["category"] = rows["item_id"].map(prod["category"])
    rows["semantic_sim"] = semantic_sim(src.users, src.products, rows["user_id"], rows["item_id"])

    clicks = src.impressions[["impression_id", "product_id"]].merge(
        attributed[attributed["type"] == "view"], on="impression_id")
    clicks = clicks.merge(src.impressions[["impression_id", "ts"]], on="impression_id")
    clicks["ts"] = clicks["ts"] + clicks["lag_s"]
    rows["hist_ctr_7d"] = hist_ctr_7d(rows, src.impressions, clicks)

    tiers = aov.daily_tiers(src.events, rows["user_id"].unique(),
                            rows["impression_ts"].min(), rows["impression_ts"].max())
    rows["day"] = (rows["impression_ts"] // aov.DAY_S).astype("int64")
    rows = rows.merge(tiers[["user_id", "day", "aov_tier"]], on=["user_id", "day"], how="left")
    rows["user_aov_tier"] = rows["aov_tier"].fillna(config.AOV_DEFAULT_TIER)

    rows["query_text"] = ""
    rows["device"] = "unknown"
    rows["bm25"] = np.nan
    rows["rating"] = np.nan

    out = rows[COLUMNS].sort_values(["qid", "displayed_pos"]).reset_index(drop=True)
    out = out.astype(DTYPES)
    report.add("rows_out", len(out))
    report.add("groups_out", out["qid"].nunique())
    report.add("sessions_out", out["session_id"].nunique())
    report.add("grade_distribution", out["relevance_grade"].value_counts().sort_index().to_dict())
    return out
