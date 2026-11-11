from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from repricer.api.app import create_app
from repricer.config import DEFAULT_POLICIES_PATH, Settings
from repricer.db.connection import connect
from repricer.db.migrate import migrate
from repricer.seed import generate, write_legacy_db
from tests.conftest import FROZEN_NOW


@pytest.fixture
def api(tmp_path: Path) -> Iterator[TestClient]:
    db = tmp_path / "api.db"
    conn = connect(db)
    write_legacy_db(conn, generate(seed=42, now=FROZEN_NOW))
    migrate(conn)
    conn.close()
    policies = tmp_path / "policies.yaml"
    shutil.copy(DEFAULT_POLICIES_PATH, policies)
    settings = Settings(
        db_path=db,
        model="claude-sonnet-5-5",
        extractor="regex",
        llm_mode="replay",
        hard_cap_bp=5000,
        llm_batch=20,
        policies_path=policies,
        cassette_dir=tmp_path / "cassettes",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client
