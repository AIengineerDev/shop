"""E2E: every CLI as a subprocess, then the API as a real HTTP server over loopback.

make_starter_db.py -> ingest (-v, JSON logs) -> ingest again (no-op) -> revalidate after a spec edit
-> scripts/serve.py -> HTTP checks -> JSON access log. No in-process shortcuts.
"""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from tests.helpers import live_server, run_cli, subprocess_env

pytestmark = pytest.mark.e2e


def _summary(stdout: str) -> dict[str, str]:
    out = {}
    for line in stdout.splitlines():
        parts = line.rsplit("  ", 1)
        if len(parts) == 2:
            out[parts[0].strip()] = parts[1].strip()
    return out


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    db = tmp_path_factory.mktemp("e2e") / "starter.sqlite"
    r = run_cli("scripts/make_starter_db.py", "--out", str(db), "--seed", "7", "--rows", "150")
    assert r.returncode == 0, r.stderr
    first = run_cli("-m", "catalog_onboard", "ingest", "--db", str(db), "-v")
    assert first.returncode == 0, first.stderr
    second = run_cli("-m", "catalog_onboard", "ingest", "--db", str(db))
    assert second.returncode == 0, second.stderr
    return db, first, second


def test_ingest_summary_logs_and_idempotency(pipeline):
    db, first, second = pipeline
    s1, s2 = _summary(first.stdout), _summary(second.stdout)
    assert s1["processed"] == "150" and s1["errors"] == "0" and s1["llm"] == "fake"
    assert int(s1["draft"]) + int(s1["quarantined"]) == 150
    assert s2["processed"] == "0" and s2["skipped (already processed)"] == "150"  # AC-3.4
    # §8: every per-item log line is JSON and carries feed_item_id + run_id
    lines = [json.loads(line) for line in first.stderr.splitlines() if line.startswith("{")]
    done = [x for x in lines if x.get("event") == "item_done"]
    assert len(done) == 150
    assert all(x["feed_item_id"] and x["run_id"] == s1["run_id"] for x in done)
    assert len({x["feed_item_id"] for x in done}) == 150


def test_live_server_serves_the_ingested_catalog(pipeline):
    db, first, _ = pipeline
    s1 = _summary(first.stdout)
    with live_server(db) as (base, logs):
        with httpx.Client(base_url=base, timeout=10) as c:
            health = c.get("/health").json()
            assert health["status"] == "ok" and health["rules_version"] == s1["rules_version"]
            stats = c.get("/stats").json()
            assert stats["draft"] == int(s1["draft"]) and stats["quarantined"] == int(s1["quarantined"])
            shorts = c.get("/catalog/draft", params={"q": "shorts", "limit": 500}).json()["items"]
            assert any(i["feed_item_id"] == "SAMPLE-0001" for i in shorts)
            trace = c.get("/items/SAMPLE-0001/trace").json()["runs"]
            assert [r["stage"] for r in trace][0] == "clean" and trace[-1]["stage"] == "route"
            nf = c.get("/catalog/draft/999999")
            assert nf.status_code == 404 and nf.json()["code"] == "NOT_FOUND"
            assert c.get("/quarantine", params={"limit": 9999}).status_code == 422
            assert c.post("/catalog/draft").status_code == 405  # AC-4.4 over the wire
            assert c.get("/docs").status_code == 200
        # Concurrent burst against real uvicorn: a thread-local SQLite connection used from the
        # wrong worker thread returned 500s only when requests overlapped (found by the load test).
        paths = ["/catalog/draft", "/quarantine", "/stats", "/catalog/draft?limit=500", "/taxonomy"]

        def burst(i: int) -> list[int]:
            with httpx.Client(base_url=base, timeout=10) as bc:
                return [bc.get(paths[(i + k) % len(paths)]).status_code for k in range(20)]

        with ThreadPoolExecutor(8) as pool:
            codes = [code for batch in pool.map(burst, range(8)) for code in batch]
        assert set(codes) == {200}, {code: codes.count(code) for code in set(codes)}
    access = [json.loads(line) for line in logs["stderr"].splitlines() if line.startswith("{")]
    paths = {(x["path"], x["status_code"]) for x in access if x.get("event") == "http_request"}
    assert ("/health", 200) in paths and ("/catalog/draft/999999", 404) in paths


def test_revalidate_cli_then_server_shows_new_outcomes(pipeline, tmp_path):
    db, _, _ = pipeline
    work = tmp_path / "work.sqlite"
    work.write_bytes(db.read_bytes())
    conn = sqlite3.connect(str(work))
    cat = conn.execute("SELECT category_id FROM draft_catalog GROUP BY category_id ORDER BY COUNT(*) DESC"
                       ).fetchone()[0]
    specs = json.loads(conn.execute("SELECT required_attributes_json FROM canonical_categories WHERE "
                                    "category_id=?", (cat,)).fetchone()[0])
    conn.execute("UPDATE canonical_categories SET required_attributes_json=? WHERE category_id=?",
                 (json.dumps([*specs, {"key": "warranty_years", "type": "int"}]), cat))
    conn.commit()
    conn.close()
    r = run_cli("-m", "catalog_onboard", "revalidate", "--db", str(work))
    assert r.returncode == 0, r.stderr
    s = _summary(r.stdout)
    assert int(s["revalidated"]) > 0 and s["errors"] == "0" and s["llm"].startswith("none")
    with live_server(work) as (base, _):
        with httpx.Client(base_url=base, timeout=10) as c:
            assert c.get("/health").json()["rules_version"] == s["rules_version"]
            assert c.get("/catalog/draft", params={"category_id": cat}).json()["items"] == []


@pytest.mark.parametrize("case", ["missing_db", "no_starter_tables", "bad_provider", "bad_rules_dir"])
def test_configuration_errors_exit_2(tmp_path: Path, case):
    """AC-3.6: non-zero exit only for configuration errors, with a message, never a traceback."""
    db = tmp_path / "x.sqlite"
    env = subprocess_env()
    if case == "no_starter_tables":
        sqlite3.connect(str(db)).close()
    elif case in ("bad_provider", "bad_rules_dir"):
        run_cli("scripts/make_starter_db.py", "--out", str(db), "--rows", "5")
        env = subprocess_env(**({"LLM_PROVIDER": "openai"} if case == "bad_provider"
                                else {"RULES_DIR": str(tmp_path / "nope")}))
    r = run_cli("-m", "catalog_onboard", "ingest", "--db", str(db), env=env)
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert r.stderr.startswith("error:") and "Traceback" not in r.stderr


def test_dry_run_writes_nothing(pipeline, tmp_path):
    db, _, _ = pipeline
    fresh = tmp_path / "fresh.sqlite"
    assert run_cli("scripts/make_starter_db.py", "--out", str(fresh), "--rows", "20").returncode == 0
    r = run_cli("-m", "catalog_onboard", "ingest", "--db", str(fresh), "--dry-run")
    assert r.returncode == 0 and _summary(r.stdout)["processed"] == "20"
    conn = sqlite3.connect(str(fresh))
    assert conn.execute("SELECT COUNT(*) FROM processed_items").fetchone()[0] == 0
