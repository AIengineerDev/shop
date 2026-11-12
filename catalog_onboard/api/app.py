"""FastAPI factory. Opens the SQLite DB read-only; no endpoint writes."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse, Response

from catalog_onboard.api.routes import build_router
from catalog_onboard.config import Settings
from catalog_onboard.db.store import Store, open_readonly
from catalog_onboard.rules import compute_rules_version, pipeline_version
from catalog_onboard.taxonomy import Taxonomy

access_log = logging.getLogger("catalog_onboard.api")


def create_app(db_path: str | Path, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    db = str(Path(db_path).resolve())
    local = threading.local()

    def get_store() -> Store:
        conn = getattr(local, "conn", None)
        if conn is None:
            conn = open_readonly(db)
            local.conn = conn
        return Store(conn)

    taxonomy = Taxonomy.from_store(get_store())

    def get_taxonomy() -> Taxonomy:
        return taxonomy

    app = FastAPI(
        title="Catalog Inspection API", version="0.1.0", description="Read-only view of the onboarding pipeline."
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

    @app.middleware("http")
    async def _access_log(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        access_log.info(
            "%s %s", request.method, request.url.path,
            extra={"event": "http_request", "method": request.method, "path": request.url.path,
                   "status_code": response.status_code,
                   "duration_ms": round((time.perf_counter() - t0) * 1000, 2)},
        )
        return response

    @app.exception_handler(HTTPException)
    async def _http_exc(_: Request, exc: HTTPException) -> JSONResponse:
        body = exc.detail if isinstance(exc.detail, dict) else {"detail": str(exc.detail), "code": "HTTP_ERROR"}
        return JSONResponse(status_code=exc.status_code, content=body)

    return app


def app_from_env() -> FastAPI:
    """Factory for multi-worker uvicorn (`scripts/serve.py --workers N`): DB path from $CATALOG_DB."""
    import os

    return create_app(os.environ["CATALOG_DB"])
