from __future__ import annotations

import importlib.util
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

from catalog_onboard.config import DEFAULT_RULES_DIR, Settings
from catalog_onboard.db.store import Store, open_readwrite
from catalog_onboard.pipeline.engine import Engine
from catalog_onboard.pipeline.llm.fake import FakeLLM
from catalog_onboard.rules import Rules, load_rules
from catalog_onboard.taxonomy import Taxonomy

REPO = Path(__file__).resolve().parents[1]
SAMPLE_ID = "SAMPLE-0001"


def _load_make_starter_db():
    spec = importlib.util.spec_from_file_location("make_starter_db", REPO / "scripts" / "make_starter_db.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_starter_db"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def make_starter_db():
    return _load_make_starter_db()


@pytest.fixture(scope="session")
def starter_db_path(tmp_path_factory: pytest.TempPathFactory, make_starter_db) -> Path:
    path = tmp_path_factory.mktemp("db") / "starter.sqlite"
    make_starter_db.build(path, seed=7, n_rows=120)
    return path


@pytest.fixture
def db_path(starter_db_path: Path, tmp_path: Path) -> Path:
    """Fresh copy of the starter DB per test."""
    dst = tmp_path / "work.sqlite"
    shutil.copy(starter_db_path, dst)
    return dst


@pytest.fixture(scope="session")
def rules() -> Rules:
    return load_rules(DEFAULT_RULES_DIR)


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_provider="fake")


@pytest.fixture
def store(db_path: Path) -> Store:
    st = Store(open_readwrite(db_path))
    st.migrate()
    return st


@pytest.fixture
def taxonomy(store: Store) -> Taxonomy:
    return Taxonomy.from_store(store)


@pytest.fixture
def engine(store: Store, taxonomy: Taxonomy, rules: Rules, settings: Settings) -> Engine:
    return Engine(store=store, taxonomy=taxonomy, rules=rules, llm=FakeLLM(), settings=settings)


@pytest.fixture
def ingested_db(db_path: Path, engine: Engine) -> Path:
    engine.run_batch()
    return db_path


def mini_taxonomy_db(path: Path, categories: list[tuple[str, str, str | None, str | None, str | None]]) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE canonical_categories (category_id TEXT PRIMARY KEY, name TEXT, parent_id TEXT,
            required_attributes_json TEXT, prohibited_keywords TEXT);
        CREATE TABLE incoming_supplier_feeds (feed_item_id TEXT PRIMARY KEY, supplier_id TEXT, raw_title TEXT,
            raw_bullets TEXT, raw_tags TEXT, raw_price TEXT, received_at TEXT);
        """
    )
    conn.executemany("INSERT INTO canonical_categories VALUES (?,?,?,?,?)", categories)
    conn.commit()
    conn.close()


# ---- no network in the test suite (spec §8) ------------------------------------------------

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}  # e2e talks to a local server


@pytest.fixture(autouse=True, scope="session")
def _block_network():
    """Fail any outbound connection to a non-loopback address. Loopback stays allowed for the
    e2e test that runs the API as a real HTTP server."""
    import socket

    real_connect = socket.socket.connect
    real_create = socket.create_connection

    def _host(address) -> str:
        return str(address[0]) if isinstance(address, tuple) else str(address)

    def guarded_connect(self, address):  # type: ignore[no-untyped-def]
        if self.family in (socket.AF_INET, socket.AF_INET6) and _host(address) not in _LOOPBACK:
            raise RuntimeError(f"network access blocked in tests: {address!r}")
        return real_connect(self, address)

    def guarded_create(address, *a, **k):  # type: ignore[no-untyped-def]
        if _host(address) not in _LOOPBACK:
            raise RuntimeError(f"network access blocked in tests: {address!r}")
        return real_create(address, *a, **k)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.create_connection = guarded_create  # type: ignore[assignment]
    yield
    socket.socket.connect = real_connect  # type: ignore[method-assign]
    socket.create_connection = real_create  # type: ignore[assignment]
