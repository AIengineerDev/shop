"""Load: AC-4.6 API latency on 10k rows and §8 ingest throughput.

- p95 < 100 ms for every list endpoint and filter on a 10k-row ingested DB (in-process)
- the same bound for a live server under concurrent clients, with zero errors
- >= 200 items/min ingest at concurrency 8 against an LLM with realistic per-call latency
  (fake LLM + LOAD_LLM_LATENCY_S, default 0.5 s), with <= 2 calls per item
"""

import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from catalog_onboard.api import create_app
from catalog_onboard.config import Settings
from catalog_onboard.db.store import Store, open_readwrite
from catalog_onboard.pipeline.engine import Engine
from catalog_onboard.pipeline.llm import LLMRequest
from catalog_onboard.pipeline.llm.fake import FakeLLM
from catalog_onboard.taxonomy import Taxonomy
from tests.helpers import live_server

pytestmark = pytest.mark.load

P95_BUDGET_S = 0.100
LLM_LATENCY_S = float(os.environ.get("LOAD_LLM_LATENCY_S", "0.5"))
THROUGHPUT_ITEMS = int(os.environ.get("LOAD_THROUGHPUT_ITEMS", "120"))
CLIENTS = int(os.environ.get("LOAD_CLIENTS", "8"))
CLIENT_SECONDS = float(os.environ.get("LOAD_CLIENT_SECONDS", "10"))
# Run the live-server test with 1 worker and with the deployment shape (several processes).
SERVER_WORKERS = int(os.environ.get("LOAD_SERVER_WORKERS", "4"))


def _p95(samples: list[float]) -> float:
    s = sorted(samples)
    return s[max(0, int(len(s) * 0.95) - 1)]


@pytest.fixture(scope="module")
def big_db(tmp_path_factory, make_starter_db, rules) -> Path:
    db = tmp_path_factory.mktemp("load") / "big.sqlite"
    make_starter_db.build(db, seed=1, n_rows=10_000)
    store = Store(open_readwrite(db))
    store.migrate()
    summary = Engine(store=store, taxonomy=Taxonomy.from_store(store), rules=rules, llm=FakeLLM(),
                     settings=Settings(llm_provider="fake")).run_batch()
    assert summary.processed == 10_000 and summary.errors == 0
    return db


def _cases(db: Path) -> dict[str, tuple[str, dict]]:
    conn = sqlite3.connect(str(db))
    sup = conn.execute("SELECT supplier_id FROM draft_catalog LIMIT 1").fetchone()[0]
    cat = conn.execute("SELECT category_id FROM draft_catalog GROUP BY category_id ORDER BY COUNT(*) DESC"
                       ).fetchone()[0]
    qcat = conn.execute("SELECT proposed_category_id FROM quarantine WHERE proposed_category_id IS NOT NULL"
                        ).fetchone()[0]
    conn.close()
    return {
        "draft": ("/catalog/draft", {}),
        "draft supplier": ("/catalog/draft", {"supplier_id": sup}),
        "draft category": ("/catalog/draft", {"category_id": cat}),
        "draft q": ("/catalog/draft", {"q": "shorts"}),
        "draft q no match": ("/catalog/draft", {"q": "zz-no-match"}),
        "draft limit 500": ("/catalog/draft", {"limit": 500}),
        "quarantine": ("/quarantine", {}),
        "quarantine supplier": ("/quarantine", {"supplier_id": sup}),
        "quarantine reason": ("/quarantine", {"reason_code": "PROHIBITED_GLOBAL"}),
        "quarantine reason no match": ("/quarantine", {"reason_code": "NO_SUCH_CODE"}),
        "quarantine category": ("/quarantine", {"proposed_category_id": qcat}),
        "stats": ("/stats", {}),
        "taxonomy": ("/taxonomy", {}),
    }


def test_ac_4_6_every_list_endpoint_p95_under_100ms(big_db: Path):
    c = TestClient(create_app(big_db))
    results = {}
    for name, (path, params) in _cases(big_db).items():
        for _ in range(3):
            c.get(path, params=params)
        samples = []
        for _ in range(40):
            t = time.perf_counter()
            r = c.get(path, params=params)
            samples.append(time.perf_counter() - t)
            assert r.status_code == 200, (name, r.text[:200])
        results[name] = _p95(samples)
    print("\nAC-4.6 p95 ms:", {k: round(v * 1000, 1) for k, v in results.items()})
    slow = {k: v for k, v in results.items() if v >= P95_BUDGET_S}
    assert not slow, slow


@pytest.mark.parametrize("workers", sorted({1, SERVER_WORKERS}))
def test_live_server_under_concurrent_clients(big_db: Path, workers: int):
    cases = list(_cases(big_db).values())
    latencies: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()
    with live_server(big_db, workers=workers) as (base, logs):
        stop_at = time.monotonic() + CLIENT_SECONDS

        def client(i: int) -> None:
            with httpx.Client(base_url=base, timeout=10) as c:
                n = i
                while time.monotonic() < stop_at:
                    path, params = cases[n % len(cases)]
                    n += 1
                    t = time.perf_counter()
                    try:
                        r = c.get(path, params=params)
                        ok = r.status_code == 200
                    except httpx.HTTPError as exc:
                        ok, r = False, exc  # type: ignore[assignment]
                    with lock:
                        latencies.append(time.perf_counter() - t)
                        if not ok:
                            errors.append(str(r))

        with ThreadPoolExecutor(max_workers=CLIENTS) as pool:
            list(pool.map(client, range(CLIENTS)))
    rps = len(latencies) / CLIENT_SECONDS
    print(f"\nlive server: {len(latencies)} requests, {rps:.0f} req/s, p95 {(_p95(latencies) * 1000):.1f} ms, "
          f"{len(errors)} errors, {CLIENTS} clients, {workers} workers; first errors: {errors[:2]}")
    server_errors = [line for line in logs.get("stderr", "").splitlines() if '"level": "ERROR"' in line]
    assert not errors, (errors[:3], server_errors[:3])
    assert len(latencies) > CLIENTS * 10
    assert _p95(latencies) < P95_BUDGET_S
    # §8: every worker process logs JSON; one access line per request served
    access = [line for line in logs["stderr"].splitlines() if '"event": "http_request"' in line]
    assert len(access) >= len(latencies), (len(access), len(latencies))


class LatencyLLM(FakeLLM):
    """FakeLLM with fixed per-call latency standing in for a real model; tracks overlap and calls."""

    def __init__(self, latency_s: float):
        self.latency_s = latency_s
        self.calls = 0
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def complete(self, request: LLMRequest) -> str:
        with self.lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.latency_s)
            return super().complete(request)
        finally:
            with self.lock:
                self.active -= 1


def test_ingest_throughput_at_concurrency_8(tmp_path: Path, make_starter_db, rules):
    db = tmp_path / "tp.sqlite"
    make_starter_db.build(db, seed=3, n_rows=THROUGHPUT_ITEMS)
    store = Store(open_readwrite(db))
    store.migrate()
    llm = LatencyLLM(LLM_LATENCY_S)
    engine = Engine(store=store, taxonomy=Taxonomy.from_store(store), rules=rules, llm=llm,
                    settings=Settings(llm_provider="fake", llm_concurrency=8))
    t0 = time.perf_counter()
    summary = engine.run_batch()
    elapsed = time.perf_counter() - t0
    per_min = summary.processed / elapsed * 60
    print(f"\ningest: {summary.processed} items in {elapsed:.1f}s = {per_min:.0f} items/min at concurrency 8, "
          f"{LLM_LATENCY_S}s/call, {llm.calls} calls ({llm.calls / summary.processed:.2f}/item), peak {llm.peak}")
    assert summary.processed == THROUGHPUT_ITEMS and summary.errors == 0
    assert llm.calls <= 2 * summary.processed  # §8 call budget
    assert llm.peak >= 6  # calls really overlap
    assert per_min >= 200, f"{per_min:.0f} items/min < 200"
