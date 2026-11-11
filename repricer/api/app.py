"""App factory. One PolicyStore per process; one SQLite connection per request."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from repricer import __version__
from repricer.api import errors
from repricer.config import Settings
from repricer.db.connection import connect
from repricer.engine.policy import PolicyStore
from repricer.extraction.llm import ClaudeExtractor
from repricer.extraction.regex_extractor import RegexExtractor
from repricer.extraction.types import Extractor


def get_settings(request: Request) -> Settings:
    s: Settings = request.app.state.settings
    return s


def get_store(request: Request) -> PolicyStore:
    s: PolicyStore = request.app.state.policy_store
    return s


def get_conn(request: Request) -> Iterator[sqlite3.Connection]:
    conn = connect(request.app.state.settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


def build_extractors(settings: Settings) -> tuple[Extractor, Extractor | None]:
    """(primary, fallback). Regex mode has no fallback; LLM mode falls back to regex."""
    regex = RegexExtractor()
    if settings.extractor == "regex":
        return regex, None
    llm = ClaudeExtractor(
        model=settings.model,
        mode=settings.llm_mode,
        cassette_dir=settings.cassette_dir if settings.llm_mode != "live" else None,
    )
    return llm, regex


SettingsDep = Annotated[Settings, Depends(get_settings)]
StoreDep = Annotated[PolicyStore, Depends(get_store)]
ConnDep = Annotated[sqlite3.Connection, Depends(get_conn)]


def create_app(settings: Settings | None = None, *, db_path: Path | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if db_path is not None:
        settings = Settings(**{**settings.__dict__, "db_path": db_path})
    app = FastAPI(title="repricer", version=__version__)
    app.state.settings = settings
    app.state.policy_store = PolicyStore(settings.policies_path)
    errors.install(app)

    from repricer.api.routes import router

    app.include_router(router, prefix="/v1")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app
