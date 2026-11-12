"""Integration: starter DB -> engine (fake LLM, concurrency 8) -> SQLite -> read-only API, in-process.

Each test crosses a real module boundary and checks what the API reports against the database.
"""

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from catalog_onboard.api import create_app
from catalog_onboard.cli import main
from catalog_onboard.config import Settings
from catalog_onboard.db.store import Store, open_readwrite
from catalog_onboard.pipeline.engine import Engine
from catalog_onboard.pipeline.llm.fake import ScriptedLLM
from catalog_onboard.taxonomy import Taxonomy

pytestmark = pytest.mark.integration


@pytest.fixture
def ingested(db_path: Path) -> Path:
    assert main(["ingest", "--db", str(db_path), "--llm", "fake", "--concurrency", "8"]) == 0
    return db_path


def _db_counts(db: Path) -> tuple[int, int]:
    conn = sqlite3.connect(str(db))
    d = conn.execute("SELECT COUNT(*) FROM draft_catalog").fetchone()[0]
    q = conn.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0]
    conn.close()
    return d, q


def _all_pages(c: TestClient, path: str, **params) -> list[dict]:
    items, cursor = [], None
    while True:
        r = c.get(path, params={**params, "limit": 7, **({"cursor": cursor} if cursor else {})})
        assert r.status_code == 200, r.text
        body = r.json()
        items += body["items"]
        cursor = body["next_cursor"]
        if not cursor:
            return items


def test_api_reflects_every_outcome_exactly_once(ingested: Path):
    c = TestClient(create_app(ingested))
    drafts, quarantined = _db_counts(ingested)
    stats = c.get("/stats").json()
    assert (stats["draft"], stats["quarantined"]) == (drafts, quarantined)
    d_items = _all_pages(c, "/catalog/draft")
    q_items = _all_pages(c, "/quarantine")
    assert len(d_items) == drafts and len({i["feed_item_id"] for i in d_items}) == drafts
    assert len(q_items) == quarantined
    assert not {i["feed_item_id"] for i in d_items} & {i["feed_item_id"] for i in q_items}
    # AC-3.1 through the API: routing invariant holds for what operators see
    assert all(all(rc["severity"] != "blocking" for rc in i["reason_codes"]) for i in d_items)
    assert all(any(rc["severity"] == "blocking" for rc in i["reason_codes"]) for i in q_items)
    per_supplier = sum(s["draft"] + s["quarantined"] for s in stats["per_supplier"])
    assert per_supplier == drafts + quarantined


def test_detail_and_trace_endpoints_agree_with_lists(ingested: Path):
    c = TestClient(create_app(ingested))
    d = c.get("/catalog/draft", params={"limit": 1}).json()["items"][0]
    detail = c.get(f"/catalog/draft/{d['listing_id']}").json()
    assert detail["listing"] == d
    trace = c.get(f"/items/{d['feed_item_id']}/trace").json()["runs"]
    assert [r["stage"] for r in detail["trace"]] == [r["stage"] for r in trace]
    assert trace[0]["stage"] == "clean" and trace[-1]["stage"] == "route"
    q = c.get("/quarantine", params={"limit": 1}).json()["items"][0]
    qd = c.get(f"/quarantine/{q['quarantine_id']}").json()
    assert qd["item"]["feed_item_id"] == q["feed_item_id"] and qd["trace"]


def test_ac_4_5_trace_of_llm_failure_includes_raw_responses(db_path: Path, rules):
    """AC-4.5 through the API: a quarantined item's trace carries the raw failed LLM output."""
    store = Store(open_readwrite(db_path))
    store.migrate()
    llm = ScriptedLLM(["not json at all", '{"category_id": "nope", "confidence": 2}'])
    Engine(store=store, taxonomy=Taxonomy.from_store(store), rules=rules, llm=llm,
           settings=Settings(llm_provider="fake", llm_max_retries=1, llm_concurrency=1)).run_batch(limit=1)
    fid = store.conn.execute("SELECT feed_item_id FROM quarantine").fetchone()[0]
    c = TestClient(create_app(db_path))
    runs = c.get(f"/items/{fid}/trace").json()["runs"]
    raw = [a["raw_response"] for r in runs if r["stage"] == "classify" for a in (r["llm_raw"] or [])]
    assert "not json at all" in raw and any("nope" in (x or "") for x in raw)
    q = c.get("/quarantine", params={"reason_code": "LLM_SCHEMA_INVALID"}).json()["items"]
    assert [i["feed_item_id"] for i in q] == [fid]


def test_revalidate_is_visible_through_the_api(ingested: Path):
    c = TestClient(create_app(ingested))
    before_version = c.get("/health").json()["rules_version"]
    cat = c.get("/catalog/draft", params={"limit": 1}).json()["items"][0]["category_id"]
    in_cat = {i["feed_item_id"] for i in _all_pages(c, "/catalog/draft", category_id=cat)}
    conn = sqlite3.connect(str(ingested))
    specs = json.loads(conn.execute("SELECT required_attributes_json FROM canonical_categories WHERE category_id=?",
                                    (cat,)).fetchone()[0])
    conn.execute("UPDATE canonical_categories SET required_attributes_json=? WHERE category_id=?",
                 (json.dumps([*specs, {"key": "warranty_years", "type": "int"}]), cat))
    conn.commit()
    conn.close()
    assert main(["revalidate", "--db", str(ingested)]) == 0

    c2 = TestClient(create_app(ingested))  # new process view: taxonomy re-read
    assert c2.get("/health").json()["rules_version"] != before_version
    assert _all_pages(c2, "/catalog/draft", category_id=cat) == []
    moved = {i["feed_item_id"] for i in _all_pages(c2, "/quarantine", reason_code="ATTR_MISSING:warranty_years")}
    # every former draft of the category moved; already-quarantined items in it gain the code too
    assert in_cat and in_cat <= moved
    node = c2.get(f"/taxonomy/{cat}").json()
    assert any(s["key"] == "warranty_years" for s in node["required_attributes"])


def test_api_is_read_only_during_and_after_ingest(ingested: Path):
    c = TestClient(create_app(ingested))
    for method in ("post", "put", "delete", "patch"):
        assert getattr(c, method)("/catalog/draft").status_code == 405
    before = _db_counts(ingested)
    for _ in range(3):
        c.get("/stats")
        c.get("/quarantine")
    assert _db_counts(ingested) == before


def test_404_and_422_contracts(ingested: Path):
    c = TestClient(create_app(ingested))
    for path in ("/catalog/draft/999999", "/quarantine/999999", "/items/NOPE/trace", "/taxonomy/no.such"):
        r = c.get(path)
        assert r.status_code == 404 and set(r.json()) == {"detail", "code"}
    assert c.get("/catalog/draft", params={"limit": 501}).status_code == 422
    assert c.get("/catalog/draft", params={"limit": 0}).status_code == 422
