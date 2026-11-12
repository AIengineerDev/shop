"""Regression tests for the API's SQLite access under concurrent requests (found by the load test)."""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from catalog_onboard.api import create_app
from catalog_onboard.db.store import SerializedConnection, Store, open_readonly


def test_fetched_cursor_semantics(ingested_db: Path):
    conn = SerializedConnection(open_readonly(ingested_db, check_same_thread=False), threading.Lock())
    cur = conn.execute("SELECT feed_item_id FROM draft_catalog ORDER BY feed_item_id LIMIT 3")
    first = cur.fetchone()
    rest = cur.fetchall()
    assert first is not None and len(rest) == 2 and cur.fetchone() is None and cur.fetchall() == []
    assert len(list(conn.execute("SELECT 1 UNION ALL SELECT 2"))) == 2
    conn.close()


def test_shared_serialized_store_from_many_threads(ingested_db: Path):
    """One connection, many threads: results identical to a single-threaded read, no errors."""
    store = Store(SerializedConnection(open_readonly(ingested_db, check_same_thread=False), threading.Lock()))
    expected = store.stats()

    def read(_):
        return store.stats(), len(store.list_drafts(supplier_id=None, category_id=None, q=None, after=None,
                                                    limit=500))

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(read, range(64)))
    assert all(r[0] == expected for r in results)
    assert len({r[1] for r in results}) == 1


def test_app_serves_concurrent_in_process_requests(ingested_db: Path):
    """Smoke test only: TestClient does not reproduce uvicorn's threading (this passed on the old
    thread-local-connection code). The real regression test for that bug is the concurrent burst
    against a live server in tests/e2e/test_cli_and_server.py."""
    app = create_app(ingested_db)
    paths = ["/catalog/draft", "/quarantine", "/stats", "/catalog/draft?limit=500", "/taxonomy"]

    def hammer(i: int) -> list[int]:
        c = TestClient(app)  # no `with`: one app lifespan, as in a real server
        return [c.get(paths[(i + k) % len(paths)]).status_code for k in range(15)]

    with ThreadPoolExecutor(8) as pool:
        codes = [code for batch in pool.map(hammer, range(8)) for code in batch]
    assert codes and set(codes) == {200}


def test_close_waits_for_in_flight_statement(ingested_db: Path):
    """close() must never run while another thread is inside a statement."""
    import time

    lock = threading.Lock()
    conn = SerializedConnection(open_readonly(ingested_db, check_same_thread=False), lock)
    lock.acquire()  # simulate a statement in flight
    closed = threading.Event()
    t = threading.Thread(target=lambda: (conn.close(), closed.set()))
    t.start()
    time.sleep(0.05)
    assert not closed.is_set()
    lock.release()
    t.join(2)
    assert closed.is_set()


def test_multi_worker_factory_reads_db_from_env_and_logs_json(ingested_db: Path, monkeypatch):
    """scripts/serve.py --workers N imports this factory in each worker process."""
    import logging

    from catalog_onboard.api.app import app_from_env
    from catalog_onboard.obs import JsonFormatter

    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    monkeypatch.setenv("CATALOG_DB", str(ingested_db))
    try:
        app = app_from_env()
        assert TestClient(app).get("/health").status_code == 200
        assert any(isinstance(h.formatter, JsonFormatter) for h in logging.getLogger().handlers)
    finally:
        root.handlers, root.level = handlers, level
