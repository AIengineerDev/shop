import json
from decimal import Decimal
from pathlib import Path

import pytest

from catalog_onboard.models import RawFeedRow
from catalog_onboard.pipeline import extract as ex
from catalog_onboard.pipeline.clean import normalize

SAMPLE = "Men's Dry-Fit Athletic Gym Shorts w/ Pockets Pack of 2 (Black/Navy, L) - Running Workout"
GOLDEN = Path(__file__).parent / "fixtures" / "normalized_golden.json"


def row(**kw) -> RawFeedRow:
    base = dict(feed_item_id="X-1", supplier_id="sup_orbit")
    base.update(kw)
    return RawFeedRow(**base)


def test_ac_1_1_sample_title(rules):
    n = normalize(row(raw_title=SAMPLE), rules)
    assert n.clean_title == "Men Dry-Fit Athletic Gym Shorts With Pockets"
    assert n.det_attrs["pack_count"] == 2
    assert set(n.det_attrs["color"]) == {"black", "navy"}
    assert n.det_attrs["size"] == ["L"]
    assert n.det_attrs["gender"] == "men"
    assert n.keyword_tail == ["Running", "Workout"]


def test_ac_1_2_bullets_dedup_and_order(rules):
    n = normalize(row(raw_title="T", raw_bullets="Alpha\n• beta;   \n alpha • Gamma"), rules)
    assert n.bullets == ["Alpha", "beta", "Gamma"]


def test_ac_1_3_tag_mapping(rules):
    mapped = normalize(row(raw_title="T", raw_tags="colour=NVY"), rules)
    assert mapped.det_attrs["color"] == ["navy"]
    assert mapped.unmapped_tags == {}

    acme = normalize(row(supplier_id="sup_acme", raw_title="T", raw_tags="clr=blk;sze=lg"), rules)
    assert acme.det_attrs["color"] == ["black"] and acme.det_attrs["size"] == ["L"]

    unknown = normalize(row(raw_title="T", raw_tags="zzz=1;colour=red"), rules)
    assert unknown.unmapped_tags == {"zzz": "1"}
    codes = [c.code for c in unknown.reason_codes]
    assert "TAG_UNMAPPED" in codes
    assert all(c.severity != "blocking" for c in unknown.reason_codes)


def test_comma_separated_tags_with_comma_values(rules):
    n = normalize(row(raw_title="T", raw_tags="size=M/XL,color=navy,red,gender=men,qty=2"), rules)
    assert n.det_attrs["size"] == ["M", "XL"]
    assert n.det_attrs["color"] == ["navy", "red"]
    assert n.det_attrs["gender"] == "men"
    assert n.det_attrs["pack_count"] == 2


def test_ac_1_5_malformed_inputs_never_raise(rules):
    n = normalize(row(raw_title="", raw_price="call for price", raw_tags="=;;=", raw_bullets=None), rules)
    codes = {c.code for c in n.reason_codes}
    assert {"TITLE_EMPTY", "PRICE_UNPARSEABLE"} <= codes
    assert n.price_min is None and n.clean_title == ""
    n2 = normalize(row(raw_title="café \x00 ﬁsh (Black)"), rules)
    assert "\x00" not in n2.clean_title and "fish" in n2.clean_title.lower()


@pytest.mark.parametrize(
    "raw,lo,hi",
    [
        ("$24.99", Decimal("24.99"), Decimal("24.99")),
        ("USD 12", Decimal("12"), Decimal("12")),
        ("$10.00 - $13.00", Decimal("10.00"), Decimal("13.00")),
        ("19,99", Decimal("19.99"), Decimal("19.99")),
    ],
)
def test_price_parse(raw, lo, hi):
    assert ex.parse_price(raw) == (lo, hi, True)


@pytest.mark.parametrize(
    "title,count",
    [("Socks Pack of 3", 3), ("Socks 6-Pack", 6), ("Socks 4pk", 4), ("Set of 12 Mugs", 12), ("Single Mug", None)],
)
def test_pack_count(title, count):
    assert ex.extract_pack_count(title)[1] == count


def test_all_caps_title_is_normalised(rules):
    n = normalize(row(raw_title="LADIES PREMIUM BUTTON-DOWN BLOUSE (BROWN, XL)"), rules)
    assert n.clean_title == "Women Premium Button-Down Blouse"
    assert n.det_attrs["size"] == ["XL"] and n.det_attrs["color"] == ["brown"]


def test_title_vs_tag_conflict_is_warning(rules):
    n = normalize(row(raw_title="Men's Tee (Red, M)", raw_tags="gender=kids"), rules)
    assert n.det_attrs["gender"] == "kids"
    assert any(c.code == "ATTR_CONFLICT:gender" and c.severity == "warning" for c in n.reason_codes)


def test_title_length_cap(rules):
    n = normalize(row(raw_title="Word " * 80), rules)
    assert len(n.clean_title) <= 150


def test_ac_1_6_no_llm_import_in_clean_stage():
    from tests.helpers import assert_modules_do_not_load

    assert_modules_do_not_load(
        ["catalog_onboard.pipeline.clean", "catalog_onboard.pipeline.extract"],
        forbidden_prefix="catalog_onboard.pipeline.llm",
    )


def test_ac_1_4_snapshot(rules, store):
    """Pure function: same rows + same rules -> identical output. Regenerate with UPDATE_GOLDEN=1."""
    import os

    rows = list(store.iter_feed_rows(limit=60))
    assert len(rows) >= 50
    out = {r.feed_item_id: normalize(r, rules).model_dump(mode="json") for r in rows}
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps({"rules_version": rules.version, "rows": out}, indent=1, sort_keys=True))
    if not GOLDEN.exists():  # never self-heal: a missing golden would make the snapshot vacuous
        pytest.fail(f"{GOLDEN} missing; generate it with UPDATE_GOLDEN=1 and review it")
    golden = json.loads(GOLDEN.read_text())
    assert golden["rules_version"] == rules.version, "rules changed: run with UPDATE_GOLDEN=1 and review the diff"
    assert golden["rows"] == out
