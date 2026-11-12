"""AC-2.4: changing a category's required_attributes_json and re-running validation without the LLM
re-evaluates outcomes from stored CandidateListing data."""

import json
import sqlite3
from pathlib import Path

from catalog_onboard.cli import main
from catalog_onboard.db.store import Store, open_readwrite
from catalog_onboard.taxonomy import Taxonomy


def _ingest(db: Path) -> None:
    assert main(["ingest", "--db", str(db), "--llm", "fake"]) == 0


def _outcomes(db: Path) -> dict[str, tuple[str, list[str]]]:
    conn = sqlite3.connect(str(db))
    out = {}
    for fid, codes in conn.execute("SELECT feed_item_id, reason_codes_json FROM draft_catalog"):
        out[fid] = ("draft", [c["code"] for c in json.loads(codes)])
    for fid, codes in conn.execute("SELECT feed_item_id, reason_codes_json FROM quarantine"):
        out[fid] = ("quarantined", [c["code"] for c in json.loads(codes)])
    conn.close()
    return out


def _category_of(db: Path, fid: str) -> str:
    conn = sqlite3.connect(str(db))
    row = conn.execute("SELECT category_id FROM draft_catalog WHERE feed_item_id=? UNION ALL "
                       "SELECT proposed_category_id FROM quarantine WHERE feed_item_id=?", (fid, fid)).fetchone()
    conn.close()
    return str(row[0])


def _set_specs(db: Path, category_id: str, specs: list[dict]) -> None:
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE canonical_categories SET required_attributes_json=? WHERE category_id=?",
                 (json.dumps(specs), category_id))
    conn.commit()
    conn.close()


def _specs(db: Path, category_id: str) -> list[dict]:
    conn = sqlite3.connect(str(db))
    raw = conn.execute("SELECT required_attributes_json FROM canonical_categories WHERE category_id=?",
                       (category_id,)).fetchone()[0]
    conn.close()
    return json.loads(raw)


def test_tightening_a_spec_moves_drafts_to_quarantine_without_llm(db_path: Path, capsys):
    _ingest(db_path)
    before = _outcomes(db_path)
    draft_fid = next(f for f, (s, _) in sorted(before.items()) if s == "draft")
    cat = _category_of(db_path, draft_fid)
    in_cat = [f for f, (s, _) in before.items() if s == "draft" and _category_of(db_path, f) == cat]
    _set_specs(db_path, cat, [*_specs(db_path, cat), {"key": "warranty_years", "type": "int", "min": 1, "max": 10}])
    capsys.readouterr()

    # The CLI wires ForbiddenLLM: any LLM call would raise and surface as an error.
    assert main(["revalidate", "--db", str(db_path)]) == 0
    out = capsys.readouterr().out
    assert "errors                             0" in out

    after = _outcomes(db_path)
    assert set(after) == set(before)  # nothing lost or duplicated
    for fid in in_cat:
        status, codes = after[fid]
        assert status == "quarantined" and "ATTR_MISSING:warranty_years" in codes
    untouched = [f for f, (s, _) in before.items() if s == "draft" and _category_of(db_path, f) != cat]
    assert untouched and all(after[f][0] == "draft" for f in untouched)


def test_relaxing_a_spec_moves_quarantined_items_to_draft(db_path: Path):
    _ingest(db_path)
    before = _outcomes(db_path)
    # an item blocked only by one missing attribute
    target = next(
        (f, [c for c in codes if c.startswith("ATTR_MISSING:")][0].split(":", 1)[1])
        for f, (s, codes) in sorted(before.items())
        if s == "quarantined" and [c for c in codes if c.startswith("ATTR_MISSING:")]
        and all(c.startswith(("ATTR_MISSING:", "ATTR_", "TAG_", "CLASSIFICATION_WEAK", "LANG_")) for c in codes)
        and len([c for c in codes if c.startswith("ATTR_MISSING:")]) == 1
    )
    fid, key = target
    cat = _category_of(db_path, fid)
    _set_specs(db_path, cat, [s for s in _specs(db_path, cat) if s["key"] != key])
    assert main(["revalidate", "--db", str(db_path)]) == 0
    status, codes = _outcomes(db_path)[fid]
    assert status == "draft" and f"ATTR_MISSING:{key}" not in codes


def test_revalidate_is_idempotent_and_traced(db_path: Path, capsys):
    _ingest(db_path)
    cat = next(iter(sorted(_outcomes(db_path))))
    cat = _category_of(db_path, cat)
    _set_specs(db_path, cat, [*_specs(db_path, cat), {"key": "eco_label", "type": "bool"}])
    assert main(["revalidate", "--db", str(db_path)]) == 0
    capsys.readouterr()
    assert main(["revalidate", "--db", str(db_path)]) == 0
    out = capsys.readouterr().out
    assert "revalidated                        0" in out  # second run: everything already at this version

    store = Store(open_readwrite(db_path))
    fid = next(iter(_outcomes(db_path)))
    stages = [r.stage for r in store.trace(fid)]
    assert "revalidate" in stages and stages[-1] == "route"
    reval = next(r for r in store.trace(fid) if r.stage == "revalidate")
    assert reval.input and reval.input["llm_calls"] == 0


def test_taxonomy_content_changes_the_pipeline_version(db_path: Path):
    store = Store(open_readwrite(db_path))
    store.migrate()
    before = Taxonomy.from_store(store).content_hash()
    cat = store.conn.execute("SELECT category_id FROM canonical_categories LIMIT 1").fetchone()[0]
    store.conn.execute("UPDATE canonical_categories SET prohibited_keywords='newbanned' WHERE category_id=?", (cat,))
    assert Taxonomy.from_store(store).content_hash() != before


def test_migrate_adds_candidate_column_to_pre_v11_databases(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE draft_catalog (listing_id INTEGER PRIMARY KEY AUTOINCREMENT, feed_item_id TEXT "
                 "NOT NULL UNIQUE, supplier_id TEXT NOT NULL, category_id TEXT NOT NULL, clean_title TEXT NOT NULL, "
                 "attributes_json TEXT NOT NULL, reason_codes_json TEXT NOT NULL DEFAULT '[]', price TEXT, "
                 "fingerprint TEXT NOT NULL, pipeline_run_id TEXT NOT NULL, rules_version TEXT NOT NULL, "
                 "created_at TEXT NOT NULL)")
    conn.commit()
    conn.close()
    store = Store(open_readwrite(db_path))
    store.migrate()
    cols = {r[1] for r in store.conn.execute("PRAGMA table_info(draft_catalog)")}
    assert "candidate_json" in cols
    _ingest(db_path)  # and the pipeline works on the migrated DB
