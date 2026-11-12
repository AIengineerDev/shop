"""Point-in-time item, seller, user and popularity-prior features (spec §5.1, §5.5).

Every aggregate for a row at time t uses only events with ts < t. `HistoryStore` holds
sorted event-time arrays keyed by item / seller / user / (user, seller) and answers
window counts with binary search, so the same store serves training and online scoring.

Popularity-prior ranks (best-seller rank in category, global percentile, top-N flag,
unpopular flag) are computed per day as of the day's start, which is point-in-time at day
granularity and cheap to precompute for serving.

Unavailable in the current sources (omitted, listed in UNAVAILABLE): item reviews
(review count, smoothed rating, seller review-weighted / top-item rating) and item price
history (purchase value always equals list price). Merchant snapshot columns are used only
for slow-moving fields (return_rate, fulfilment_days); review_score / popularity are
present-day and excluded (leakage, spec §5.1).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shopfeed import config

DAY_S = 86400.0
UNAVAILABLE = [
    "item_review_count", "item_rating_smoothed", "seller_top_item_rating",
    "seller_review_weighted_rating", "price_vs_item_median_90d", "price_is_item_min_90d",
]
HISTORY_FEATURES = [
    "item_purchases_30d", "item_purchases_all", "item_wishlist_30d", "item_age_days",
    "seller_return_rate", "seller_fulfilment_days", "seller_item_count", "seller_purchases_30d",
    "user_follows_seller", "user_order_count", "is_new_user", "user_aov_value",
    "item_bestseller_rank_category", "item_bestseller_pct_global", "is_category_top20",
    "is_unpopular",
]


def _index(keys: pd.Series, ts: pd.Series) -> dict:
    out: dict = defaultdict(lambda: np.empty(0))
    for k, v in pd.DataFrame({"k": keys.to_numpy(), "t": ts.to_numpy()}).groupby("k")["t"]:
        out[k] = np.sort(v.to_numpy())
    return out


def _value_index(keys: pd.Series, ts: pd.Series, values: pd.Series) -> dict:
    """key -> (sorted ts, cumulative value sum) for windowed means."""
    out: dict = {}
    f = pd.DataFrame({"k": keys.to_numpy(), "t": ts.to_numpy(), "v": values.to_numpy()})
    for k, g in f.sort_values("t").groupby("k"):
        out[k] = (g["t"].to_numpy(), np.concatenate([[0.0], np.cumsum(g["v"].to_numpy())]))
    return out


def _key_groups(keys):
    """Yield (key, row indices) per distinct non-null key, via one stable sort.

    Avoids pandas groupby, whose fixed overhead dominates on a 200-row request."""
    codes, uniques = pd.factorize(pd.Series(keys).to_numpy(dtype=object), use_na_sentinel=True)
    if not len(codes):
        return
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for a, b in zip(starts, ends, strict=True):
        c = sorted_codes[a]
        if c >= 0:
            yield uniques[c], order[a:b]


def _window_mean(index: dict, keys, t: np.ndarray, window_s: float) -> np.ndarray:
    out = np.full(len(t), np.nan)
    for k, idx in _key_groups(keys):
        if k not in index:
            continue
        ts, csum = index[k]
        hi = np.searchsorted(ts, t[idx], "left")
        lo = np.searchsorted(ts, t[idx] - window_s, "left")
        n = hi - lo
        with np.errstate(invalid="ignore", divide="ignore"):
            out[idx] = np.where(n > 0, (csum[hi] - csum[lo]) / np.maximum(n, 1), np.nan)
    return out


def _count(index: dict, keys, t: np.ndarray, window_s: float | None) -> np.ndarray:
    """Number of events per key in [t - window, t) (or (-inf, t) when window is None)."""
    out = np.zeros(len(t), dtype=np.int64)
    for k, idx in _key_groups(keys):
        arr = index.get(k)
        if arr is None or not len(arr):
            continue
        hi = np.searchsorted(arr, t[idx], "left")
        lo = 0 if window_s is None else np.searchsorted(arr, t[idx] - window_s, "left")
        out[idx] = hi - lo
    return out


@dataclass
class HistoryStore:
    purchases_by_item: dict
    saves_by_item: dict
    purchases_by_seller: dict
    purchases_by_user: dict
    purchase_values_by_user: dict
    follows_by_user_seller: dict
    impressions_by_item: dict
    clicks_by_item: dict
    products: pd.DataFrame  # index product_id: merchant_id, category, published_at
    merchants: pd.DataFrame  # index merchant_id: return_rate, fulfilment_days
    daily_popularity: pd.DataFrame  # (product_id, day) -> popularity-prior columns

    @classmethod
    def from_sources(cls, impressions: pd.DataFrame, events: pd.DataFrame,
                     products: pd.DataFrame, merchants: pd.DataFrame,
                     days: range | None = None) -> HistoryStore:
        p = events[events["type"] == "purchase"]
        s = events[events["type"] == "save"]
        f = events[events["type"] == "follow"]
        v = events[(events["type"] == "view") & events["impression_id"].notna()]
        prod = products.set_index("product_id")[["merchant_id", "category", "published_at"]]
        store = cls(
            purchases_by_item=_index(p["product_id"], p["ts"]),
            saves_by_item=_index(s["product_id"], s["ts"]),
            purchases_by_seller=_index(p["merchant_id"], p["ts"]),
            purchases_by_user=_index(p["user_id"], p["ts"]),
            purchase_values_by_user=_value_index(p["user_id"], p["ts"], p["value"].fillna(0.0)),
            follows_by_user_seller=_index(f["user_id"] + "|" + f["merchant_id"], f["ts"]),
            impressions_by_item=_index(impressions["product_id"], impressions["ts"]),
            clicks_by_item=_index(v["product_id"], v["ts"]),
            products=prod,
            merchants=merchants.set_index("merchant_id")[["return_rate", "fulfilment_days"]],
            daily_popularity=pd.DataFrame(),
        )
        if days is None:
            d0, d1 = int(impressions["ts"].min() // DAY_S), int(impressions["ts"].max() // DAY_S)
            days = range(d0, d1 + 1)
        store.daily_popularity = store._popularity(days)
        store._build_lookups()
        return store

    POP_COLUMNS = ("item_bestseller_rank_category", "item_bestseller_pct_global",
                   "is_category_top20", "is_unpopular")

    def _build_lookups(self) -> None:
        """Integer-keyed (product code, day) table for O(log n) popularity lookups."""
        self._prod_code = {pid: i for i, pid in enumerate(self.products.index)}
        dp = self.daily_popularity
        codes = dp["product_id"].map(self._prod_code).to_numpy(dtype=np.int64)
        keys = codes * 1_000_000 + dp["day"].to_numpy(dtype=np.int64)
        order = np.argsort(keys)
        self._pop_keys = keys[order]
        self._pop_values = {c: dp[c].to_numpy(dtype=float)[order] for c in self.POP_COLUMNS}
        self._published = self.products["published_at"].to_dict()
        self._published_by_seller = _index(self.products["merchant_id"],
                                           self.products["published_at"])

    def _popularity_lookup(self, items, day: np.ndarray) -> dict:
        codes = np.array([self._prod_code.get(i, -1) for i in items], dtype=np.int64)
        keys = codes * 1_000_000 + day
        pos = np.searchsorted(self._pop_keys, keys)
        pos_c = np.minimum(pos, len(self._pop_keys) - 1)
        found = (codes >= 0) & (len(self._pop_keys) > 0) & (self._pop_keys[pos_c] == keys)
        return {c: np.where(found, v[pos_c], np.nan) for c, v in self._pop_values.items()}

    def _popularity(self, days: range) -> pd.DataFrame:
        items = self.products.index.to_numpy()
        cats = self.products["category"].to_numpy()
        frames = []
        for day in days:
            t = np.full(len(items), day * DAY_S)
            buys = _count(self.purchases_by_item, items, t, 30 * DAY_S)
            imps = _count(self.impressions_by_item, items, t, 30 * DAY_S)
            clks = _count(self.clicks_by_item, items, t, 30 * DAY_S)
            ctr = np.divide(clks, imps, out=np.zeros(len(items)), where=imps > 0)
            unpop = (buys == 0) & (imps >= config.UNPOPULAR_MIN_IMPRESSIONS) & (
                ctr < config.UNPOPULAR_MAX_CTR)
            f = pd.DataFrame({"product_id": items, "category": cats, "buys": buys,
                              "is_unpopular": unpop.astype("int64")})
            # Ties broken by product_id so the rank is deterministic.
            f = f.sort_values(["category", "buys", "product_id"], ascending=[True, False, True])
            f["item_bestseller_rank_category"] = f.groupby("category").cumcount() + 1
            f["item_bestseller_pct_global"] = f["buys"].rank(pct=True, method="average")
            f["is_category_top20"] = ((f["item_bestseller_rank_category"] <= config.CATEGORY_TOP_N)
                                      & (f["buys"] > 0)).astype("int64")
            f["day"] = day
            frames.append(f.drop(columns=["category", "buys"]))
        return pd.concat(frames, ignore_index=True)

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        """rows needs: item_id, seller_id, user_id, impression_ts. Returns HISTORY_FEATURES."""
        t = rows["impression_ts"].to_numpy(dtype=float)
        item = rows["item_id"].to_numpy(dtype=object)
        seller = rows["seller_id"].to_numpy(dtype=object)
        user = rows["user_id"].to_numpy(dtype=object)
        out: dict[str, np.ndarray] = {}
        out["item_purchases_30d"] = _count(self.purchases_by_item, item, t, 30 * DAY_S)
        out["item_purchases_all"] = _count(self.purchases_by_item, item, t, None)
        out["item_wishlist_30d"] = _count(self.saves_by_item, item, t, 30 * DAY_S)
        published = np.array([self._published.get(i, np.nan) for i in item], dtype=float)
        out["item_age_days"] = np.clip((t - published) / DAY_S, 0, None)
        m = self.merchants.reindex(seller)
        out["seller_return_rate"] = m["return_rate"].to_numpy()
        out["seller_fulfilment_days"] = m["fulfilment_days"].to_numpy()
        out["seller_item_count"] = _count(self._published_by_seller, seller, t, None)
        out["seller_purchases_30d"] = _count(self.purchases_by_seller, seller, t, 30 * DAY_S)
        pair = np.array([f"{u}|{s}" for u, s in zip(user, seller, strict=True)], dtype=object)
        follows = _count(self.follows_by_user_seller, pair, t, None)
        out["user_follows_seller"] = (follows > 0).astype("int64")
        out["user_order_count"] = _count(self.purchases_by_user, user, t, None)
        out["is_new_user"] = (out["user_order_count"] == 0).astype("int64")
        out["user_aov_value"] = _window_mean(self.purchase_values_by_user, user, t,
                                             config.AOV_LOOKBACK_S)
        out.update(self._popularity_lookup(item, (t // DAY_S).astype(np.int64)))
        return pd.DataFrame({c: out[c] for c in HISTORY_FEATURES}, index=rows.index)

    def item_impressions(self, items, t: np.ndarray, window_s: float = 30 * DAY_S) -> np.ndarray:
        """Impressions per item in [t - window, t). Serving-rule input, not a model feature."""
        return _count(self.impressions_by_item, items, np.asarray(t, dtype=float), window_s)
