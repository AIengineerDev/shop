import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from catalog_onboard.api import create_app
from tests.conftest import SAMPLE_ID


@pytest.fixture
def client(ingested_db: Path) -> TestClient:
    return TestClient(create_app(ingested_db))


def test_health(client: TestClient, rules):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["rules_version"] == rules.version


def test_ac_4_1_openapi_has_no_any_typed_fields(client: TestClient):
    spec = client.get("/openapi.json").json()
    empty = []

    def walk(o, path):
        if isinstance(o, dict):
            if o == {}:
                empty.append(path)
            for k, v in o.items():
                walk(v, f"{path}/{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

    walk(spec["components"]["schemas"], "")
    assert empty == []
    assert client.get("/docs").status_code == 200


def test_ac_4_2_pagination_is_stable(client: TestClient):
    seen = []
    cursor = None
    for _ in range(100):
        r = client.get("/catalog/draft", params={"limit": 7, **({"cursor": cursor} if cursor else {})})
        assert r.status_code == 200
        body = r.json()
        seen.extend(i["listing_id"] for i in body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break
    assert seen == sorted(seen) and len(seen) == len(set(seen))
    total = client.get("/stats").json()["draft"]
    assert len(seen) == total
    assert client.get("/catalog/draft", params={"limit": 501}).status_code == 422
    assert client.get("/catalog/draft", params={"limit": 0}).status_code == 422
    assert client.get("/catalog/draft", params={"cursor": "!!bad"}).status_code == 422


def test_filters(client: TestClient):
    r = client.get("/catalog/draft", params={"supplier_id": "sup_acme", "q": "shorts"}).json()
    assert r["items"] and all(
        i["supplier_id"] == "sup_acme" and "shorts" in i["clean_title"].lower() for i in r["items"]
    )
    r = client.get("/quarantine", params={"reason_code": "PROHIBITED_GLOBAL"}).json()
    assert r["items"] and all(
        any(c["code"].startswith("PROHIBITED_GLOBAL") for c in i["reason_codes"]) for i in r["items"]
    )


def test_ac_4_3_not_found_shape(client: TestClient):
    for url in ("/catalog/draft/999999", "/quarantine/999999", "/items/NOPE/trace", "/taxonomy/nope.nope"):
        r = client.get(url)
        assert r.status_code == 404
        assert set(r.json()) == {"detail", "code"} and r.json()["code"] == "NOT_FOUND"


def test_detail_endpoints_include_trace(client: TestClient):
    first = client.get("/catalog/draft", params={"limit": 1}).json()["items"][0]
    d = client.get(f"/catalog/draft/{first['listing_id']}").json()
    assert d["listing"]["feed_item_id"] == first["feed_item_id"]
    assert [r["stage"] for r in d["trace"]][0] == "clean"
    q = client.get("/quarantine", params={"limit": 1}).json()["items"][0]
    qd = client.get(f"/quarantine/{q['quarantine_id']}").json()
    assert qd["item"]["reason_codes"] and qd["trace"]


def test_ac_4_5_trace_contains_llm_raw(client: TestClient):
    runs = client.get(f"/items/{SAMPLE_ID}/trace").json()["runs"]
    cls = next(r for r in runs if r["stage"] == "classify")
    assert cls["llm_raw"] and "raw_response" in cls["llm_raw"][0]


def test_taxonomy_endpoints(client: TestClient):
    tree = client.get("/taxonomy").json()
    assert tree["leaf_count"] > 0 and any(n["is_leaf"] for n in tree["nodes"])
    node = client.get("/taxonomy/kitchen.cutlery.chef_knives").json()
    assert "switchblade" in node["effective_prohibited_keywords"] and node["is_leaf"]


def test_stats(client: TestClient):
    s = client.get("/stats").json()
    assert s["draft"] > 0 and s["quarantined"] > 0 and s["top_reason_codes"] and s["per_supplier"]


def test_ac_4_4_api_opens_db_readonly(client: TestClient, ingested_db: Path):
    before = sqlite3.connect(str(ingested_db)).execute("SELECT COUNT(*) FROM draft_catalog").fetchone()[0]
    for m in ("post", "put", "delete", "patch"):
        assert getattr(client, m)("/catalog/draft").status_code == 405
    after = sqlite3.connect(str(ingested_db)).execute("SELECT COUNT(*) FROM draft_catalog").fetchone()[0]
    assert before == after


@pytest.mark.load
def test_ac_4_6_list_latency_p95_under_100ms(tmp_path: Path, make_starter_db):
    from catalog_onboard.cli import main

    db = tmp_path / "big.sqlite"
    make_starter_db.build(db, seed=1, n_rows=10_000)
    assert main(["ingest", "--db", str(db), "--llm", "fake"]) == 0
    c = TestClient(create_app(db))
    times = []
    for _ in range(40):
        t = time.perf_counter()
        assert c.get("/catalog/draft", params={"limit": 50, "supplier_id": "sup_acme"}).status_code == 200
        times.append(time.perf_counter() - t)
    times.sort()
    assert times[int(len(times) * 0.95) - 1] < 0.1
