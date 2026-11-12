"""C5: LLM-bound stages run concurrently; outcomes are identical to a sequential run (§8)."""

import shutil
import sqlite3
import threading
import time
import zlib
from pathlib import Path

from catalog_onboard.config import Settings
from catalog_onboard.db.store import Store, open_readwrite
from catalog_onboard.pipeline.engine import Engine
from catalog_onboard.pipeline.llm import LLMRequest
from catalog_onboard.pipeline.llm.fake import FakeLLM
from catalog_onboard.taxonomy import Taxonomy


class SlowFake(FakeLLM):
    """FakeLLM with fixed latency; records peak concurrent calls."""

    def __init__(self, delay_s: float):
        self.delay_s = delay_s
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def complete(self, request: LLMRequest) -> str:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay_s)
            return super().complete(request)
        finally:
            with self.lock:
                self.active -= 1


def _engine(db: Path, rules, llm, concurrency: int) -> Engine:
    store = Store(open_readwrite(db))
    store.migrate()
    settings = Settings(llm_provider="fake", llm_concurrency=concurrency)
    return Engine(store=store, taxonomy=Taxonomy.from_store(store), rules=rules, llm=llm, settings=settings)


def _snapshot(db: Path) -> dict:
    conn = sqlite3.connect(str(db))
    out = {}
    for fid, cat, attrs, codes in conn.execute(
        "SELECT feed_item_id, category_id, attributes_json, reason_codes_json FROM draft_catalog"
    ):
        out[fid] = ("draft", cat, attrs, codes)
    for fid, cat, codes in conn.execute(
        "SELECT feed_item_id, proposed_category_id, reason_codes_json FROM quarantine"
    ):
        out[fid] = ("quarantined", cat, None, codes)
    stages = {
        fid: [s for (s,) in conn.execute(
            "SELECT stage FROM pipeline_runs WHERE feed_item_id=? ORDER BY seq", (fid,))]
        for fid in out
    }
    conn.close()
    return {"outcomes": out, "stages": stages}


def test_concurrent_run_equals_sequential_run(starter_db_path: Path, tmp_path: Path, rules):
    seq_db, par_db = tmp_path / "seq.sqlite", tmp_path / "par.sqlite"
    shutil.copy(starter_db_path, seq_db)
    shutil.copy(starter_db_path, par_db)
    s1 = _engine(seq_db, rules, FakeLLM(), concurrency=1).run_batch()
    s8 = _engine(par_db, rules, SlowFake(0.002), concurrency=8).run_batch()
    assert (s1.processed, s1.draft, s1.quarantined, s1.errors) == (s8.processed, s8.draft, s8.quarantined, s8.errors)
    a, b = _snapshot(seq_db), _snapshot(par_db)
    assert a["outcomes"] == b["outcomes"]  # incl. DUPLICATE_LISTING, which depends on input order
    assert a["stages"] == b["stages"]


def test_llm_calls_overlap_and_batch_is_faster(db_path: Path, rules, tmp_path: Path):
    other = tmp_path / "other.sqlite"
    shutil.copy(db_path, other)
    slow1, slow8 = SlowFake(0.02), SlowFake(0.02)
    t0 = time.perf_counter()
    _engine(db_path, rules, slow1, concurrency=1).run_batch(limit=40)
    sequential = time.perf_counter() - t0
    t0 = time.perf_counter()
    _engine(other, rules, slow8, concurrency=8).run_batch(limit=40)
    parallel = time.perf_counter() - t0
    assert slow1.peak == 1 and slow8.peak >= 4
    assert parallel < sequential / 3, (sequential, parallel)


def test_worker_failure_is_isolated(db_path: Path, rules):
    class FlakyFake(FakeLLM):
        def complete(self, request: LLMRequest) -> str:
            if zlib.crc32(request.user.encode()) % 5 == 0:  # deterministic ~20% failures
                raise RuntimeError("worker blew up")
            return super().complete(request)

    eng = _engine(db_path, rules, FlakyFake(), concurrency=8)
    summary = eng.run_batch(limit=50)
    assert summary.processed == 50 and 0 < summary.errors < 50
    conn = sqlite3.connect(str(db_path))
    codes = [c for (c,) in conn.execute("SELECT reason_codes_json FROM quarantine")]
    assert any("PIPELINE_ERROR:classify" in c for c in codes)
    total = conn.execute("SELECT (SELECT COUNT(*) FROM draft_catalog) + (SELECT COUNT(*) FROM quarantine)").fetchone()
    assert total[0] == 50
