"""Structured logging: feed_item_id and run_id on every line (spec §8)."""

import io
import json
import logging

import pytest
from fastapi.testclient import TestClient

from catalog_onboard import obs
from catalog_onboard.api import create_app
from catalog_onboard.pipeline.engine import Engine


@pytest.fixture(autouse=True)
def _restore_root_logger():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers, root.level = handlers, level


def _capture():
    buf = io.StringIO()
    handler = obs.configure_logging(logging.INFO, stream=buf)
    return buf, handler


def _lines(buf):
    return [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]


def test_context_is_stamped_and_reset():
    buf, _ = _capture()
    log = logging.getLogger("t")
    with obs.item_context("run1", "ITEM-1"):
        log.info("inside", extra={"event": "x"})
    log.info("outside")
    inside, outside = _lines(buf)
    assert inside["feed_item_id"] == "ITEM-1" and inside["run_id"] == "run1" and inside["event"] == "x"
    assert outside["feed_item_id"] is None and outside["run_id"] is None
    assert set(outside) >= {"ts", "level", "logger", "msg", "feed_item_id", "run_id"}


def test_exceptions_are_serialised():
    buf, _ = _capture()
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("t").exception("failed")
    (line,) = _lines(buf)
    assert "ValueError: boom" in line["exc"]


def test_engine_logs_one_line_per_item_with_ids(engine: Engine):
    buf, _ = _capture()
    summary = engine.run_batch(limit=15)
    done = [r for r in _lines(buf) if r.get("event") == "item_done"]
    assert len(done) == summary.processed == 15
    assert all(r["feed_item_id"] and r["run_id"] == summary.run_id and r["status"] for r in done)
    assert len({r["feed_item_id"] for r in done}) == 15


def test_api_access_log_line(ingested_db):
    buf, _ = _capture()
    c = TestClient(create_app(ingested_db))
    assert c.get("/health").status_code == 200
    (line,) = [r for r in _lines(buf) if r.get("event") == "http_request"]
    assert line["method"] == "GET" and line["path"] == "/health" and line["status_code"] == 200
    assert line["duration_ms"] >= 0
