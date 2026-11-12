"""Load: open-loop traffic against the real Ranker (spec §12.4, §12.5, §13.4).

Requests arrive on a fixed schedule (LOAD_RPS for LOAD_SECONDS) and are served by a small
worker pool; latency is measured from the SCHEDULED arrival, so queueing counts. Requests
carry 200 candidates (MAX_CANDIDATES), the worst case. The production timeout (30 ms)
and rules are on, so a slow ranker shows up as fallbacks, not just as slow tests.

Gates:
- 0 exceptions escape; every response is a permutation of its candidates
- fallback rate <= 0.1% (§12.5 fallback budget)
- end-to-end p99 <= 50 ms (§12.5 SLA), ranking p50 <= 21 ms budget (§12.4)
- model scoring p50 <= 2 ms, p99 <= 8 ms (§1 ship gate)
- achieved throughput >= 95% of offered
- page 2+ from cache: hit rate >= 95%, p99 <= 10 ms
- chaos: 10% injected feature-store failures degrade to fallback, never raise
"""

from __future__ import annotations

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from shopfeed import config
from shopfeed.serving.pagination import Paginator
from shopfeed.serving.ranker import Ranker

pytestmark = pytest.mark.load

RPS = float(os.environ.get("LOAD_RPS", "25"))
SECONDS = float(os.environ.get("LOAD_SECONDS", "40"))
WORKERS = int(os.environ.get("LOAD_WORKERS", "2"))
SLA_P99_MS = 50.0
RANKING_P50_BUDGET_MS = 21.0


@pytest.fixture(scope="module")
def requests_200(synth_dataset):
    te, _ = synth_dataset.part("test")
    qids = list(te["qid"].unique())
    reqs = []
    for i in range(0, len(qids) - 10, 3):
        chunk = qids[i:i + 12]
        r = te[te["qid"].isin(chunk)].drop_duplicates("item_id").head(config.MAX_CANDIDATES)
        reqs.append(r.assign(qid=int(chunk[0])).reset_index(drop=True))
    assert reqs and all(len(r) >= 150 for r in reqs)
    return reqs


def _open_loop(fn, payloads, rps, seconds, workers):
    n = int(rps * seconds)
    start = time.perf_counter() + 0.05
    schedule = [start + i / rps for i in range(n)]
    results = [None] * n

    def job(i):
        delay = schedule[i] - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        try:
            out = fn(payloads[i % len(payloads)])
            err = None
        except Exception as e:  # noqa: BLE001 - counted, must be zero
            out, err = None, e
        results[i] = (time.perf_counter() - schedule[i], out, err)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(job, range(n)))
    wall = time.perf_counter() - start
    return results, n / wall


def test_sustained_load_meets_sla(synth_dataset, synth_model, requests_200):
    ranker = Ranker.load(synth_model["dir"], synth_dataset.store)  # production timeout
    for r in requests_200[:5]:
        ranker.rank(r)  # warm-up
    ranker.counters = {"requests": 0, "fallback": 0}

    results, achieved = _open_loop(ranker.rank, requests_200, RPS, SECONDS, WORKERS)
    n = len(results)
    errors = [e for _, _, e in results if e is not None]
    assert not errors, errors[:3]
    for i, (_, res, _) in enumerate(results):
        assert sorted(res.item_ids) == sorted(requests_200[i % len(requests_200)]["item_id"])

    e2e = np.array([lat for lat, _, _ in results]) * 1e3
    timings = pd.DataFrame([r.timings_ms for _, r, _ in results if r.timings_ms])
    fallbacks = sum(r.fallback for _, r, _ in results)
    summary = {
        "requests": n, "offered_rps": RPS, "achieved_rps": round(achieved, 1),
        "fallbacks": fallbacks, "e2e_p50": np.percentile(e2e, 50),
        "e2e_p99": np.percentile(e2e, 99), "ranking_p50": timings["total"].median(),
        "scoring_p50": timings["scoring"].median(),
        "scoring_p99": np.percentile(timings["scoring"], 99),
    }
    print("\nLOAD", {k: round(float(v), 2) for k, v in summary.items()})

    assert fallbacks <= max(1, int(0.001 * n)), summary
    assert summary["e2e_p99"] <= SLA_P99_MS, summary
    assert summary["ranking_p50"] <= RANKING_P50_BUDGET_MS, summary
    assert summary["scoring_p50"] <= 2.0 and summary["scoring_p99"] <= 8.0, summary
    assert achieved >= 0.95 * RPS, summary


def test_pagination_under_load(synth_dataset, synth_model, requests_200):
    ranker = Ranker.load(synth_model["dir"], synth_dataset.store, timeout_s=1.0)
    by_qid = {int(r["qid"].iloc[0]): r for r in requests_200}
    p = Paginator(lambda q: ranker.rank(by_qid[q]).item_ids, page_size=20)
    cursors = [p.first_page(q).next_cursor for q in by_qid]
    latencies = []
    for _ in range(9):  # pages 2..10 for every query
        nxt = []
        for c in cursors:
            if c is None:
                continue
            t0 = time.perf_counter()
            page = p.next_page(c)
            latencies.append((time.perf_counter() - t0) * 1e3)
            nxt.append(page.next_cursor)
        cursors = nxt
    assert p.cache.hit_rate() >= 0.95
    assert np.percentile(latencies, 99) <= 10.0
    for q in by_qid:
        assert len(p.served[q]) == len(set(p.served[q]))


def test_chaos_store_failures_degrade_not_raise(synth_dataset, synth_model, requests_200):
    ranker = Ranker.load(synth_model["dir"], synth_dataset.store, timeout_s=1.0)
    real_store, rng = ranker.store, random.Random(config.SEED)
    injected = 0

    class FlakyStore:
        def __getattr__(self, name):
            nonlocal injected
            if name == "features" and rng.random() < 0.10:
                injected += 1
                raise ConnectionError("feature store unavailable")
            return getattr(real_store, name)

    ranker.store = FlakyStore()
    results = [ranker.rank(requests_200[i % len(requests_200)]) for i in range(300)]
    assert injected > 0
    assert sum(r.fallback for r in results) == injected
    assert all(len(r.item_ids) == len(set(r.item_ids)) > 0 for r in results)
