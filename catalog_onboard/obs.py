"""Structured JSON logging with per-item context (spec §8).

Every log line carries `feed_item_id` and `run_id`: the engine binds them in a contextvar for the
duration of each item, and a handler-level filter stamps them on every record (null outside an
item), including records from third-party loggers. Use `configure_logging()` once per process
(CLI, API server); library code only calls `logging.getLogger(...)`.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO

_FEED_ITEM_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("feed_item_id", default=None)
_RUN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)

# Optional structured fields passed via `extra=` that the formatter emits when present.
EXTRA_FIELDS = ("event", "status", "stage", "codes", "duration_ms", "method", "path", "status_code",
                "processed", "draft", "quarantined", "errors", "skipped")


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "feed_item_id", None) is None:
            record.feed_item_id = _FEED_ITEM_ID.get()
        if getattr(record, "run_id", None) is None:
            record.run_id = _RUN_ID.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "feed_item_id": getattr(record, "feed_item_id", None),
            "run_id": getattr(record, "run_id", None),
        }
        for key in EXTRA_FIELDS:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


@contextmanager
def run_context(run_id: str) -> Iterator[None]:
    token = _RUN_ID.set(run_id)
    try:
        yield
    finally:
        _RUN_ID.reset(token)


@contextmanager
def item_context(run_id: str, feed_item_id: str) -> Iterator[None]:
    run_token = _RUN_ID.set(run_id)
    item_token = _FEED_ITEM_ID.set(feed_item_id)
    try:
        yield
    finally:
        _FEED_ITEM_ID.reset(item_token)
        _RUN_ID.reset(run_token)


def configure_logging(level: int = logging.WARNING, stream: IO[str] | None = None) -> logging.Handler:
    """Install one JSON handler on the root logger (idempotent: replaces previous handlers)."""
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(ContextFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    return handler
