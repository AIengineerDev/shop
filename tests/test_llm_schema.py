import json

import pytest
from pydantic import ValidationError

from catalog_onboard.models import AttributeSpec
from catalog_onboard.pipeline.llm import prompts
from catalog_onboard.pipeline.llm.schema import (
    build_attribute_model,
    build_classification_model,
    json_schema_for,
)


def test_classification_literal_rejects_out_of_set():
    m = build_classification_model(["a.b", "a.c"])
    ok = m.model_validate({"category_id": "a.c", "confidence": 0.5, "rationale": "x"})
    assert ok.category_id == "a.c"
    with pytest.raises(ValidationError):
        m.model_validate({"category_id": "a.zzz", "confidence": 0.5, "rationale": "x"})
    with pytest.raises(ValidationError):
        m.model_validate({"category_id": "a.b", "confidence": 1.5, "rationale": "x"})
    with pytest.raises(ValidationError):
        m.model_validate({"category_id": "a.b", "confidence": 0.5, "rationale": "x", "extra": 1})


def test_attribute_model_mirrors_spec():
    specs = [
        AttributeSpec(key="size", type="enum", allowed=["S", "M"], multi=True),
        AttributeSpec(key="gender", type="enum", allowed=["men", "women"]),
        AttributeSpec(key="pack_count", type="int", min=1, max=10),
        AttributeSpec(key="capacity_oz", type="float", min=1, max=100),
        AttributeSpec(key="material", type="string", max_len=5, pattern="^[a-z]+$"),
        AttributeSpec(key="organic", type="bool"),
    ]
    m = build_attribute_model(specs)
    good = m.model_validate(
        {"size": ["S"], "gender": "men", "pack_count": 2, "capacity_oz": 12.5, "material": "wool", "organic": True}
    )
    assert good.model_dump()["size"] == ["S"]
    assert m.model_validate({}).model_dump() == {
        k: None for k in ("size", "gender", "pack_count", "capacity_oz", "material", "organic")
    }
    for bad in (
        {"size": "S"},
        {"gender": "kids"},
        {"pack_count": 11},
        {"capacity_oz": 0},
        {"material": "toolong"},
        {"material": "Wool"},
        {"unknown": 1},
    ):
        with pytest.raises(ValidationError):
            m.model_validate(bad)


def test_json_schema_is_strict():
    m = build_attribute_model([AttributeSpec(key="gender", type="enum", allowed=["men"])])
    schema = json_schema_for(m)
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["gender"]
    json.dumps(schema)  # serialisable


def test_ac_2_6_prompts_contain_no_compliance_vocabulary(taxonomy, rules):
    from catalog_onboard.models import Candidate, NormalizedListing

    listing = NormalizedListing(
        feed_item_id="x",
        supplier_id="s",
        clean_title="Chef Knife",
        bullets=["Sharp"],
        tags={},
        unmapped_tags={},
        det_attrs={},
        keyword_tail=[],
        price_min=None,
        price_max=None,
        reason_codes=[],
        raw_json={},
    )
    cands = [Candidate(category_id=c, path=taxonomy.path(c), score=1.0) for c in taxonomy.leaf_ids[:5]]
    text = " ".join(
        [
            prompts.CLASSIFY_SYSTEM,
            prompts.EXTRACT_SYSTEM,
            prompts.classify_user(listing, cands),
            prompts.extract_user(listing, taxonomy.required_attributes("kitchen.cutlery.chef_knives")),
        ]
    ).lower()
    for word in ("prohibit", "complian", "policy", "banned", "allowed to sell", "quarantine"):
        assert word not in text
    for kw in rules.global_prohibited:
        assert kw not in text
    for cid in taxonomy.leaf_ids:
        for kw in taxonomy.effective_prohibited(cid):
            assert kw not in text
