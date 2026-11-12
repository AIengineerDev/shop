"""FastAPI factory. Opens the SQLite DB read-only; no endpoint writes."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from catalog_onboard.api.routes import build_router
from catalog_onboard.config import Settings
from catalog_onboard.db.store import SerializedConnection, Store, open_readonly
from catalog_onboard.rules import compute_rules_version, pipeline_version
from catalog_onboard.taxonomy import Taxonomy

access_log = logging.getLogger("catalog_onboard.api")
# One lock per process for all SQLite work in the API (see SerializedConnection).
DB_LOCK = threading.Lock()


class AccessLogMiddleware:
    """Pure-ASGI access log: one JSON line per request (spec §8). Cheaper under concurrency than
    @app.middleware("http") (BaseHTTPMiddleware), which wraps every response body."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        t0 = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            method, path = scope.get("method", ""), scope.get("path", "")
            access_log.info(
                "%s %s", method, path,
                extra={"event": "http_request", "method": method, "path": path, "status_code": status["code"],
                       "duration_ms": round((time.perf_counter() - t0) * 1000, 2)},
            )


def create_app(db_path: str | Path, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    db = str(Path(db_path).resolve())

    # One read-only connection per process, shared by all request threads. Every statement runs
    # under DB_LOCK (SerializedConnection), so cross-thread use is strictly serial and
    # check_same_thread=False is safe. Found by the load test:
    #  - a thread-local connection was used from the wrong thread (FastAPI resolves sync
    #    dependencies and runs sync endpoints on different worker threads) -> 500s
    #  - unserialised concurrent sqlite3 reads hit a GIL convoy (36 req/s on 8 threads)
    # A single serialised connection fixes both and needs no per-request open/close.
    conn = open_readonly(db, check_same_thread=False)
    store = Store(SerializedConnection(conn, DB_LOCK))

    def get_store() -> Store:
        return store

    taxonomy = Taxonomy.from_store(store)

    def get_taxonomy() -> Taxonomy:
        return taxonomy

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        store.conn.close()  # SerializedConnection.close: waits for any in-flight statement

    app = FastAPI(
        title="Catalog Inspection API",
        version="0.1.0",
        description="Read-only view of the onboarding pipeline.",
        lifespan=lifespan,
    )
    app.include_router(
        build_router(
            get_store,
            get_taxonomy,
            # same version string the engine stamps on outcomes: rule files + taxonomy content
            rules_version=pipeline_version(compute_rules_version(settings.rules_dir), taxonomy.content_hash()),
            db_path=db,
        )
    )

    app.add_middleware(AccessLogMiddleware)

    @app.exception_handler(HTTPException)
    async def _http_exc(_: Request, exc: HTTPException) -> JSONResponse:
        body = exc.detail if isinstance(exc.detail, dict) else {"detail": str(exc.detail), "code": "HTTP_ERROR"}
        return JSONResponse(status_code=exc.status_code, content=body)

    return app


def app_from_env() -> FastAPI:
    """Factory for multi-worker uvicorn (`scripts/serve.py --workers N`): DB path from $CATALOG_DB.

    Runs in each worker process, so it also installs JSON logging there (spec §8): without it,
    worker processes emitted no access log at all."""
    import os

    from catalog_onboard.obs import configure_logging

    configure_logging(logging.INFO)
    return create_app(os.environ["CATALOG_DB"])
