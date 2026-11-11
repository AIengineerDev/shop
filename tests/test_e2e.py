"""End-to-end: build DB -> ingest with the fake LLM -> inspect via API."""

from pathlib import Path

from fastapi.testclient import TestClient

from catalog_onboard.api import create_app
from catalog_onboard.cli import main
from tests.conftest import SAMPLE_ID


def test_sample_end_to_end(tmp_path: Path, make_starter_db):
    db = tmp_path / "e2e.sqlite"
    make_starter_db.build(db, seed=7, n_rows=80)
    assert main(["ingest", "--db", str(db), "--llm", "fake"]) == 0

    client = TestClient(create_app(db))
    drafts = client.get("/catalog/draft", params={"q": "gym shorts", "supplier_id": "sup_acme"}).json()["items"]
    sample = next(d for d in drafts if d["feed_item_id"] == SAMPLE_ID)
    assert sample["category_id"] == "apparel.men.bottoms.shorts"
    assert sample["attributes"]["gender"] == "men"
    assert sample["attributes"]["size"] == ["L"]
    assert sorted(sample["attributes"]["color"]) == ["black", "navy"]
    assert sample["attributes"]["pack_count"] == 2
    assert sample["price"] == "24.99"

    stats = client.get("/stats").json()
    assert stats["draft"] + stats["quarantined"] == 80
    assert not any(c["code"].startswith("PIPELINE_ERROR") for c in stats["top_reason_codes"])
