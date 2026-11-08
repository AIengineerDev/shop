"""Request path: context -> candidates -> light -> heavy -> value -> rerank -> log.

`FeedService` holds the in-memory state a serving replica would hold: catalog, batch feature
state (refreshed by `refresh_batch`), streaming store, ANN index, user embedding cache, and the
impression logger. All time comes from the request (`now`) so replay and tests are deterministic.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from shopfeed.config import DAY, HEADS, Config
from shopfeed.features.batch import BatchState, compute_batch_state
from shopfeed.features.snapshot import build_features
from shopfeed.features.streaming import StreamingStore
from shopfeed.ranking.heavy import RankerArtifacts
from shopfeed.ranking.light import LightRanker
from shopfeed.ranking.value import value_score
from shopfeed.rerank.policy import rerank
from shopfeed.retrieval.ann import BruteForceIndex
from shopfeed.retrieval.sources import UserContext, eligibility_mask, enrich_products, generate_candidates
from shopfeed.retrieval.two_tower import RetrievalArtifacts
from shopfeed.schema import FeedCard, FeedRequest, FeedResponse, Impression
from shopfeed.serving.logger import ImpressionLogger
from shopfeed.features.batch import AFF_COLS, PROFILE_COLS

ORDER_STATUS_WINDOW_S = 5 * DAY


@dataclass
class _Timer:
    t: Dict[str, float]

    def mark(self, name: str, start: float) -> float:
        self.t[name] = round((time.perf_counter() - start) * 1000, 2)
        return time.perf_counter()


class FeedService:
    def __init__(self, tables: Dict[str, pd.DataFrame], retrieval: RetrievalArtifacts, light: LightRanker,
                 heavy: RankerArtifacts, cfg: Config, now: float, logger: Optional[ImpressionLogger] = None,
                 streaming: Optional[StreamingStore] = None, seed: int = 0, policy: str = "model") -> None:
        """policy: "model" ranks by the value score; "random" shuffles eligible candidates (a control arm
        / logging policy for A/B and IPS tests). Eligibility, rerank rules and logging apply to both."""
        if policy not in ("model", "random"):
            raise ValueError("policy must be model or random")
        self.policy = policy
        self.cfg = cfg
        self.tables = tables
        self.retrieval, self.light, self.heavy = retrieval, light, heavy
        self.model_version = heavy.version if policy == "model" else f"{heavy.version}+random_policy"
        self.products = enrich_products(tables["products"], tables["merchants"])
        self._prod_by_id = self.products.set_index("product_id")
        self.users = tables["users"].set_index("user_id")
        self.follows: Dict[str, Set[str]] = tables["follows"].groupby("user_id").merchant_id.apply(set).to_dict()
        self.streaming = streaming or StreamingStore()
        if streaming is None:
            self.streaming.add_events(tables["events"], now=now)
        self.index = BruteForceIndex().build(retrieval.item_ids, retrieval.item_vecs)
        self.logger = logger or ImpressionLogger()
        self.rng = np.random.default_rng(seed)
        self.shown: Dict[str, Dict[str, float]] = {}
        self.batch: Optional[BatchState] = None
        self.user_vecs: Dict[str, np.ndarray] = {}
        self.popularity = pd.Series(dtype=float)
        self.refresh_batch(now)

    # ------------------------------------------------------------------ state
    def refresh_batch(self, as_of: float) -> None:
        """The 'daily' job: recompute batch features and user embeddings as of `as_of`."""
        self.batch = compute_batch_state(self.tables["events"], self.tables["impressions"], self.tables["users"],
                                         self.tables["products"], self.tables["merchants"], as_of)
        vecs = self.retrieval.embed_users(self.batch.users)
        self.user_vecs = dict(zip(self.batch.users.user_id.values, vecs))
        self.popularity = self.batch.products.set_index("product_id").n_purchases
        prior = self.batch.users.iloc[0:1].copy()
        prior[AFF_COLS] = 1.0 / len(AFF_COLS); prior[PROFILE_COLS] = 0.0
        prior[["n_purchases", "n_views_7d", "shop_pay_history"]] = 0.0
        self._prior_vec = self.retrieval.embed_users(prior)[0]
        self._priorrow_idx = prior.assign(user_id="__prior__")

    def add_products(self, new: pd.DataFrame) -> int:
        """Publish path: catalog write + content embedding + index insert, in one step. A product is
        retrievable through the fresh lane and the two-tower index before it has a single interaction."""
        new = new[~new.product_id.isin(self.tables["products"].product_id)]
        if len(new) == 0:
            return 0
        self.tables["products"] = pd.concat([self.tables["products"], new[self.tables["products"].columns]], ignore_index=True)
        self.products = enrich_products(self.tables["products"], self.tables["merchants"])
        self._prod_by_id = self.products.set_index("product_id")
        self.index.add(new.product_id.values, self.retrieval.embed_items(new))
        return len(new)

    def ingest_events(self, events: pd.DataFrame, now: float) -> Dict[str, int]:
        res = self.streaming.add_events(events, now=now)
        if res["accepted"]:
            keep = events[~events.event_id.isin(self.tables["events"].event_id)] if "event_id" in events else events
            keep = keep.reindex(columns=self.tables["events"].columns)
            if len(keep):
                self.tables["events"] = pd.concat([self.tables["events"], keep], ignore_index=True)
        return res

    # ------------------------------------------------------------------ context
    def build_context(self, user_id: str, country: str, now: float) -> UserContext:
        known = user_id in self.users.index
        blocked = set(self.users.loc[user_id, "blocked_merchants"]) if known else set()
        followed = set(self.follows.get(user_id, set()))
        rb = self.cfg.eligibility.repeat_block_s
        purchases = self.streaming.user_last_ts(user_id, "purchase")
        hidden = set(self.streaming.user_last_ts(user_id, "hide")) | set(self.streaming.user_last_ts(user_id, "not_interested"))
        recent_shown = {p for p, t in self.shown.get(user_id, {}).items() if t >= now - rb}
        cats = set(self._prod_by_id.reindex(list(purchases)).category.dropna()) if purchases else set()
        has_history = bool(purchases) or bool(followed) or (known and bool(self.users.loc[user_id, "shop_pay_history"]))
        uvec = self.user_vecs.get(user_id)
        if uvec is None or not has_history:
            uvec = self.user_vecs.get(user_id, self._prior_vec)
        return UserContext(
            user_id=user_id, country=country, now=now, blocked_merchants=blocked, followed_merchants=followed,
            recent_purchased={p for p, t in purchases.items() if t >= now - rb}, recent_shown=recent_shown,
            hidden=hidden, purchase_last_ts=purchases, purchased_categories=cats, user_vec=uvec, is_cold=not has_history,
        )

    # ------------------------------------------------------------------ fallback
    def _fallback(self, ctx: UserContext) -> pd.DataFrame:
        """Popular + fresh items passing only hard safety filters (no personalization)."""
        safe_ctx = UserContext(user_id=ctx.user_id, country=ctx.country, now=ctx.now, blocked_merchants=ctx.blocked_merchants,
                               hidden=ctx.hidden)
        mask = eligibility_mask(self.products, safe_ctx, self.cfg.eligibility)
        sel = self.products[mask].copy()
        if len(sel) == 0:
            return sel.assign(score=pd.Series(dtype=float), source="fallback", sources=None)
        pop = self.popularity.reindex(sel.product_id).fillna(0).values
        fresh = ((ctx.now - sel.published_at.values) <= DAY).astype(float)
        sel["score"] = np.log1p(pop) / (np.log1p(pop).max() or 1.0) + fresh
        sel["source"] = "fallback"
        sel["sources"] = [["fallback"]] * len(sel)
        return sel.sort_values("score", ascending=False).head(self.cfg.rerank.light_cut)[["product_id", "score", "source", "sources"]]

    # ------------------------------------------------------------------ request
    def recommend(self, req: FeedRequest) -> FeedResponse:
        now = req.now if req.now is not None else time.time()
        timings: Dict[str, float] = {}
        T = _Timer(timings)
        t = time.perf_counter()
        request_id = f"req_{uuid.uuid4().hex[:12]}"

        ctx = self.build_context(req.user_id, req.country, now)
        t = T.mark("context", t)

        cands = generate_candidates(ctx, self.products, self.index, self.streaming, self.popularity,
                                    self.cfg.caps, self.cfg.eligibility, self.cfg.rerank.fresh_window_s)
        fallback = False
        if len(cands) == 0:
            cands = self._fallback(ctx)
            fallback = True
        t = T.mark("candidates", t)
        if len(cands) == 0:
            T.mark("total", t)
            return FeedResponse(request_id=request_id, user_id=req.user_id, model_version=self.model_version,
                                fallback=True, cards=[], timings_ms=timings)

        pairs = pd.DataFrame({"user_id": req.user_id, "product_id": cands.product_id.values, "ts": now, "position": 0})
        batch = self.batch
        if req.user_id not in set(batch.users.user_id.values):
            batch = BatchState(batch.as_of, pd.concat([batch.users, self._priorrow_idx.assign(user_id=req.user_id)], ignore_index=True), batch.products, batch.merchants)
        X = build_features(pairs, batch, self.streaming, self.tables["products"], self.tables["follows"])
        t = T.mark("features", t)

        cut = self.light.cut(cands, X, self.cfg.rerank.light_cut)
        Xc = X[cut.row_idx.values]
        t = T.mark("light", t)

        P = self.heavy.predict_proba(Xc)
        prod = cut[["product_id"]].merge(self.products, on="product_id", how="left")
        cut = cut.reset_index(drop=True)
        cut["value"] = value_score(P, prod.price.values, self.cfg.value) if self.policy == "model" else self.rng.random(len(cut))
        for i, h in enumerate(HEADS):
            cut[f"p_{h}"] = P[:, i]
        cut["merchant_id"] = prod.merchant_id.values
        cut["category"] = prod.category.values
        cut["published_at"] = prod.published_at.values
        cut["price"] = prod.price.values
        t = T.mark("heavy", t)

        page = rerank(cut, self.cfg.rerank, now, self.rng)
        page = page.merge(cut[["product_id", "price"] + [f"p_{h}" for h in HEADS] + ["row_idx"]], on="product_id", how="left")
        t = T.mark("rerank", t)

        cards, imps = [], []
        for r in page.itertuples(index=False):
            imp_id = f"imp_{uuid.uuid4().hex[:12]}"
            feats = X[int(r.row_idx)].tolist() if r.row_idx == r.row_idx else []
            scores = {h: float(getattr(r, f"p_{h}")) for h in HEADS}
            imps.append(Impression(impression_id=imp_id, request_id=request_id, user_id=req.user_id, product_id=r.product_id,
                                   card_type="product", ts=now, position=int(r.position), propensity=float(r.propensity),
                                   model_version=self.model_version, source=str(r.source), scores=scores, value=float(r.value), features=feats))
            cards.append(FeedCard(impression_id=imp_id, product_id=r.product_id, merchant_id=r.merchant_id, card_type="product",
                                  category=r.category, price=float(r.price), position=int(r.position), source=str(r.source),
                                  value=float(r.value), is_exploration=bool(r.is_exploration), reason=str(r.reason)))
            self.shown.setdefault(req.user_id, {})[r.product_id] = now
        self.logger.log(imps)
        T.mark("log", t)
        timings["total"] = round(sum(v for k, v in timings.items() if k != "total"), 2)
        return FeedResponse(request_id=request_id, user_id=req.user_id, model_version=self.model_version,
                            fallback=fallback, cards=cards, timings_ms=timings)

    def in_flight_orders(self, user_id: str, now: float) -> List[str]:
        """Order-status cards: purchases in the last 5 days without a return. Utility, never ranked."""
        p = self.streaming.user_last_ts(user_id, "purchase")
        r = self.streaming.user_last_ts(user_id, "return")
        return sorted(pid for pid, t in p.items() if t >= now - ORDER_STATUS_WINDOW_S and r.get(pid, -1) < t)
