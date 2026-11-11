"""FastAPI factory. Opens the SQLite DB read-only; no endpoint writes."""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from catalog_onboard.api.routes import build_router
from catalog_onboard.config import Settings
from catalog_onboard.db.store import Store, open_readonly
from catalog_onboard.rules import compute_rules_version
from catalog_onboard.taxonomy import Taxonomy


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
        build_router(get_store, get_taxonomy, rules_version=compute_rules_version(settings.rules_dir), db_path=db)
    )

    @app.exception_handler(HTTPException)
    async def _http_exc(_: Request, exc: HTTPException) -> JSONResponse:
        body = exc.detail if isinstance(exc.detail, dict) else {"detail": str(exc.detail), "code": "HTTP_ERROR"}
        return JSONResponse(status_code=exc.status_code, content=body)

    return app
