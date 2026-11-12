"""Online ranker (spec §12, M8): load artefact, build features with the training code path,
score, apply §5.5 rules, and degrade instead of failing.

Fallback: any exception, or a ranking path slower than config.RANK_TIMEOUT_S, returns the
fallback order: BM25 descending, then the incoming (retrieval) order for ties / NaN BM25.
Python cannot pre-empt a running call, so the timeout is checked after scoring; the caller's
RPC deadline remains the hard stop.

GC: `Ranker.load(..., freeze_gc=True)` runs gc.collect() + gc.freeze() once the model and
HistoryStore are in memory. Without it, full (gen-2) collections walk the store's object graph
and pause the process for ~50 ms, which the load test showed as timeout fallbacks (spec §12.5
budget 0.1%). Frozen objects are never scanned; call it again after a store reload.

Every request appends its feature rows to `request_log` (a sink with `.append(dict)`) so
train-serve skew can be measured by recomputing features offline (§13.4).
"""

from __future__ import annotations

import gc
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from shopfeed import config
from shopfeed.features.history import HistoryStore
from shopfeed.features.ranking import build_features
from shopfeed.serving.rules import RuleFlags, apply_rules

REQUEST_LOG_MAXLEN = 1000


@dataclass
class RankResult:
    item_ids: list[str]
    scores: list[float]
    fallback: bool
    reason: str = ""
    timings_ms: dict = field(default_factory=dict)
    model_version: str = ""


def fallback_order(candidates: pd.DataFrame) -> list[str]:
    bm25 = candidates["bm25"].astype(float).fillna(-np.inf).to_numpy()
    order = np.argsort(-bm25, kind="stable")
    return candidates["item_id"].to_numpy()[order].tolist()


class Ranker:
    def __init__(self, booster: lgb.Booster, feature_names: list[str], store: HistoryStore,
                 model_version: str = "", flags: RuleFlags | None = None,
                 timeout_s: float = config.RANK_TIMEOUT_S, request_log=None):
        self.booster = booster
        self.feature_names = feature_names
        self.store = store
        self.model_version = model_version
        self.flags = flags or RuleFlags()
        self.timeout_s = timeout_s
        # Bounded by default: an unbounded in-process log is a memory leak. Production should
        # pass a sink that ships records off-process (append-only interface).
        self.request_log = request_log if request_log is not None else deque(
            maxlen=REQUEST_LOG_MAXLEN)
        self.counters = {"requests": 0, "fallback": 0}

    @classmethod
    def load(cls, artefact_dir: Path, store: HistoryStore, freeze_gc: bool = True,
             **kw) -> Ranker:
        artefact_dir = Path(artefact_dir)
        booster = lgb.Booster(model_file=str(artefact_dir / "model.txt"))
        names = json.loads((artefact_dir / "features.json").read_text())["features"]
        version = ""
        meta = artefact_dir / "meta.json"
        if meta.exists():
            version = json.loads(meta.read_text()).get("git_sha", "")[:12]
        ranker = cls(booster, names, store, model_version=version, **kw)
        if freeze_gc:
            gc.collect()
            gc.freeze()
        return ranker

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X[self.feature_names].to_numpy(dtype=np.float64),
                                    num_threads=config.SCORING_THREADS)

    def rank(self, candidates: pd.DataFrame) -> RankResult:
        """candidates: CSV-shaped rows for ONE request (one qid), in retrieval order,
        at most config.MAX_CANDIDATES."""
        self.counters["requests"] += 1
        t0 = time.perf_counter()
        timings = {}
        try:
            dup = candidates["item_id"].duplicated()
            if dup.any():  # retrieval should not emit duplicates; never show one twice
                self.counters["duplicate_candidates"] = (
                    self.counters.get("duplicate_candidates", 0) + int(dup.sum()))
                candidates = candidates[~dup]
            if len(candidates) > config.MAX_CANDIDATES:
                candidates = candidates.iloc[: config.MAX_CANDIDATES]
            cand = candidates.reset_index(drop=True)
            X, _ = build_features(cand, self.store)
            t1 = time.perf_counter()
            scores = self.score(X)
            t2 = time.perf_counter()
            ranked = apply_rules(cand.assign(score=scores, **self._rule_inputs(cand, X)),
                                 self.flags)
            t3 = time.perf_counter()
            timings = {"features": (t1 - t0) * 1e3, "scoring": (t2 - t1) * 1e3,
                       "rules": (t3 - t2) * 1e3, "total": (t3 - t0) * 1e3}
            # numpy arrays, not Python lists: one object per request instead of ~8K floats
            # for the garbage collector to walk.
            self.request_log.append({"qid": int(cand["qid"].iloc[0]),
                                     "model_version": self.model_version,
                                     "item_ids": cand["item_id"].to_numpy(dtype=object),
                                     "features": X[self.feature_names].to_numpy()})
            if t3 - t0 > self.timeout_s:
                return self._fallback(candidates, "timeout", timings)
            return RankResult(ranked["item_id"].tolist(), ranked["score"].tolist(), False,
                              timings_ms=timings, model_version=self.model_version)
        except Exception as e:  # noqa: BLE001 - any failure must degrade, never 5xx
            return self._fallback(candidates, f"error: {type(e).__name__}: {e}", timings)

    def _rule_inputs(self, cand: pd.DataFrame, X: pd.DataFrame) -> dict:
        t = cand["impression_ts"].to_numpy(dtype=float)
        return {"is_unpopular": X["is_unpopular"].to_numpy(),
                "item_age_days": X["item_age_days"].to_numpy(),
                "item_bestseller_rank_category": X["item_bestseller_rank_category"].to_numpy(),
                "is_category_top20": X["is_category_top20"].to_numpy(),
                "item_impressions_30d": self.store.item_impressions(cand["item_id"], t)}

    def _fallback(self, candidates, reason, timings) -> RankResult:
        self.counters["fallback"] += 1
        ids = fallback_order(candidates)
        return RankResult(ids, [float("nan")] * len(ids), True, reason, timings,
                          self.model_version)
