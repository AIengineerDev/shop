import json
from pathlib import Path

import pytest

from catalog_onboard.db.store import Store, open_readonly, open_readwrite
from catalog_onboard.taxonomy import Taxonomy, TaxonomyConfigError, parse_attribute_specs, parse_prohibited
from tests.conftest import mini_taxonomy_db


def test_leaf_and_path(taxonomy: Taxonomy):
    assert taxonomy.is_leaf("apparel.men.bottoms.shorts")
    assert not taxonomy.is_leaf("apparel.men.bottoms")
    assert not taxonomy.is_leaf("does.not.exist")
    assert taxonomy.path("apparel.men.bottoms.shorts") == "Apparel > Men's Clothing > Men's Bottoms > Shorts"
    assert "apparel.men.bottoms" not in taxonomy.leaf_ids


def test_prohibited_keywords_are_inherited(taxonomy: Taxonomy):
    eff = taxonomy.effective_prohibited("kitchen.cutlery.chef_knives")
    assert "switchblade" in eff and "butterfly knife" in eff
    eff_kids = taxonomy.effective_prohibited("apparel.kids.pajamas")
    assert eff_kids == ("counterfeit", "replica", "choking hazard", "flammable")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("a,b", ("a", "b")),
        ("a|B|a", ("a", "b")),
        (" a ; b\nc ", ("a", "b", "c")),
        (None, ()),
        ("", ()),
    ],
)
def test_parse_prohibited(raw, expected):
    assert parse_prohibited(raw) == expected


def test_unknown_spec_field_fails_fast():
    with pytest.raises(TaxonomyConfigError, match="invalid attribute spec"):
        parse_attribute_specs("x", json.dumps([{"key": "size", "type": "enum", "allowed": ["S"], "bogus": 1}]))


def test_enum_without_allowed_fails_fast():
    with pytest.raises(TaxonomyConfigError):
        parse_attribute_specs("x", json.dumps([{"key": "size", "type": "enum"}]))


def test_bad_json_fails_fast():
    with pytest.raises(TaxonomyConfigError, match="not valid JSON"):
        parse_attribute_specs("x", "{nope")


def test_missing_parent_fails_fast(tmp_path: Path):
    db = tmp_path / "t.sqlite"
    mini_taxonomy_db(db, [("a.b", "B", "a", None, None)])
    with pytest.raises(TaxonomyConfigError, match="parent"):
        Taxonomy.from_store(Store(open_readwrite(db)))


def test_readonly_connection_rejects_writes(ingested_db: Path):
    st = Store(open_readonly(ingested_db))
    import sqlite3

    with pytest.raises(sqlite3.OperationalError):
        st.conn.execute("DELETE FROM draft_catalog")
