import json
from pathlib import Path

from catalog_onboard.config import Settings
from catalog_onboard.db.store import Store
from catalog_onboard.models import RawFeedRow
from catalog_onboard.pipeline.engine import Engine
from catalog_onboard.pipeline.llm import LLMRequest
from catalog_onboard.pipeline.llm.fake import FakeLLM, ScriptedLLM
from catalog_onboard.rules import Rules
from catalog_onboard.taxonomy import Taxonomy
from tests.conftest import SAMPLE_ID


def q_codes(store: Store, feed_item_id: str) -> list[str]:
    row = store.conn.execute(
        "SELECT reason_codes_json FROM quarantine WHERE feed_item_id=?", (feed_item_id,)
    ).fetchone()
    return [c["code"] for c in json.loads(row[0])] if row else []


def test_ac_2_1_sample_lands_in_draft(engine: Engine, store: Store):
    summary = engine.run_batch(limit=1)
    assert summary.processed == 1 and summary.draft == 1 and summary.errors == 0
    d = store.list_drafts(supplier_id=None, category_id=None, q=None, after=None, limit=10)[0]
    assert d.feed_item_id == SAMPLE_ID
    assert d.category_id == "apparel.men.bottoms.shorts"
    assert d.attributes["gender"] == "men"
    assert d.attributes["size"] == ["L"]
    assert sorted(d.attributes["color"]) == ["black", "navy"]
    assert d.attributes["pack_count"] == 2
    assert d.attributes["material"] == "polyester"
    assert not any(c.severity == "blocking" for c in d.reason_codes)
    stages = [r.stage for r in store.trace(SAMPLE_ID)]
    assert stages[0] == "clean" and stages[-1] == "route"


def test_ac_3_4_rerun_is_noop_and_new_rules_supersede(
    engine: Engine, store: Store, taxonomy: Taxonomy, rules: Rules, settings
):
    first = engine.run_batch(limit=20)
    runs_before = store.conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
    second = engine.run_batch(limit=20)
    assert first.processed == 20 and second.processed == 0 and second.skipped == 20
    assert store.conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0] == runs_before

    new_rules = Rules(
        abbreviations=rules.abbreviations,
        supplier_tag_keys=rules.supplier_tag_keys,
        supplier_tag_values=rules.supplier_tag_values,
        global_prohibited=rules.global_prohibited,
        version="v2",
    )
    eng2 = Engine(store=store, taxonomy=taxonomy, rules=new_rules, llm=FakeLLM(), settings=settings)
    third = eng2.run_batch(limit=20)
    assert third.processed == 20
    # one outcome row per item, old trail retained
    assert (
        store.conn.execute("SELECT COUNT(*) FROM draft_catalog").fetchone()[0]
        + store.conn.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0]
        == 20
    )
    assert store.conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0] > runs_before
    assert [tuple(r) for r in store.conn.execute("SELECT DISTINCT rules_version FROM draft_catalog")] == [("v2",)]


def test_ac_3_5_bad_row_is_quarantined_not_fatal(store: Store, taxonomy: Taxonomy, rules: Rules, settings):
    class Exploding:
        name = "exploding"

        def complete(self, request: LLMRequest) -> str:
            raise RuntimeError("kaboom")  # not an LLMTransportError: unexpected failure

    eng = Engine(store=store, taxonomy=taxonomy, rules=rules, llm=Exploding(), settings=settings)
    summary = eng.run_batch(limit=3)
    assert summary.processed == 3 and summary.errors == 3 and summary.quarantined == 3
    codes = q_codes(store, SAMPLE_ID)
    assert codes == ["PIPELINE_ERROR:classify"]
    assert any(r.stage == "error" for r in store.trace(SAMPLE_ID))


def test_ac_2_2_llm_schema_failure_quarantines_with_raw(store: Store, taxonomy: Taxonomy, rules: Rules):
    settings = Settings(llm_provider="fake", llm_max_retries=1)
    llm = ScriptedLLM(["garbage", '{"category_id": "nope", "confidence": 1, "rationale": ""}'])
    eng = Engine(store=store, taxonomy=taxonomy, rules=rules, llm=llm, settings=settings)
    eng.run_batch(limit=1)
    assert "LLM_SCHEMA_INVALID:classify" in q_codes(store, SAMPLE_ID)
    trace = store.trace(SAMPLE_ID)
    cls = next(r for r in trace if r.stage == "classify")
    assert cls.llm_raw and cls.llm_raw[0]["raw_response"] == "garbage"
    assert "nope" in cls.llm_raw[1]["raw_response"]


def test_classification_disagree_policy(store: Store, taxonomy: Taxonomy, rules: Rules):
    settings = Settings(llm_provider="fake", llm_max_retries=0, high_conf_threshold=0.5, disagree_quarantine_below=0.8)
    # low-confidence LLM pick that differs from the deterministic top-1 -> blocking
    low = ScriptedLLM(
        [json.dumps({"category_id": "apparel.men.bottoms.jeans", "confidence": 0.3, "rationale": "r"}), "{}"]
    )
    Engine(store=store, taxonomy=taxonomy, rules=rules, llm=low, settings=settings).run_batch(limit=1)
    assert "CLASSIFICATION_DISAGREE" in q_codes(store, SAMPLE_ID)

    # same disagreement with high confidence -> warning, LLM pick wins, supersedes under new rules version
    hi_rules = Rules(**{**rules.__dict__, "version": "x"})
    hi = ScriptedLLM(
        [json.dumps({"category_id": "apparel.men.bottoms.jeans", "confidence": 0.95, "rationale": "r"}), "{}"]
    )
    Engine(store=store, taxonomy=taxonomy, rules=hi_rules, llm=hi, settings=settings).run_batch(limit=1)
    d = store.list_drafts(supplier_id=None, category_id=None, q=None, after=None, limit=5)
    assert d and d[0].category_id == "apparel.men.bottoms.jeans"
    assert any(c.code == "CLASSIFICATION_DISAGREE" and c.severity == "warning" for c in d[0].reason_codes)


def test_ac_2_3_non_leaf_never_assigned(ingested_db: Path, store: Store, taxonomy: Taxonomy):
    rows = store.conn.execute("SELECT category_id FROM draft_catalog").fetchall()
    assert rows and all(taxonomy.is_leaf(r[0]) for r in rows)


def test_ac_3_1_no_blocking_in_draft_no_clean_in_quarantine(ingested_db: Path, store: Store):
    for (codes_json,) in store.conn.execute("SELECT reason_codes_json FROM draft_catalog"):
        assert not any(c["severity"] == "blocking" for c in json.loads(codes_json))
    for (codes_json,) in store.conn.execute("SELECT reason_codes_json FROM quarantine"):
        assert any(c["severity"] == "blocking" for c in json.loads(codes_json))


def test_injected_defects_are_caught(ingested_db: Path, store: Store):
    rows = store.conn.execute(
        "SELECT f.feed_item_id, f.injected_defect, q.reason_codes_json FROM incoming_supplier_feeds f "
        "LEFT JOIN quarantine q USING(feed_item_id) WHERE f.injected_defect IN "
        "('empty_title','prohibited_global','prohibited_category','bad_price')"
    ).fetchall()
    assert rows
    for fid, defect, codes_json in rows:
        assert codes_json is not None, f"{fid} ({defect}) reached draft"
        codes = {c["code"].split(":")[0] for c in json.loads(codes_json)}
        expected = {
            "empty_title": {"TITLE_EMPTY", "TITLE_TOO_SHORT"},
            "prohibited_global": {"PROHIBITED_GLOBAL"},
            "prohibited_category": {"PROHIBITED_KEYWORD"},
            "bad_price": {"PRICE_INVALID"},
        }[defect]
        assert codes & expected, f"{fid} ({defect}) codes={codes}"


def test_duplicate_detection(engine: Engine, store: Store):
    engine.run_batch(limit=1)
    dup = RawFeedRow(
        feed_item_id="DUP-1",
        supplier_id="sup_acme",
        raw_title="Men's Dry-Fit Athletic Gym Shorts w/ Pockets Pack of 2 (Black/Navy, L) - Running Workout",
        raw_bullets="Material: polyester",
        raw_tags="sze=L;clr=blk,nvy;qty=2;sex=M",
        raw_price="$20",
    )
    out = engine.process_item(dup, run_id="t")
    assert out.status == "quarantined" and "DUPLICATE_LISTING" in [c.code for c in out.reason_codes]


def test_dry_run_writes_nothing(store: Store, taxonomy: Taxonomy, rules: Rules, settings):
    eng = Engine(store=store, taxonomy=taxonomy, rules=rules, llm=FakeLLM(), settings=settings, dry_run=True)
    s = eng.run_batch(limit=5)
    assert s.processed == 5
    for t in ("draft_catalog", "quarantine", "pipeline_runs", "processed_items"):
        assert store.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0
