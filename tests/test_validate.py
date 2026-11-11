import pytest

from catalog_onboard.models import AttributeSpec
from catalog_onboard.pipeline.reconcile import merge_attributes, missing_specs
from catalog_onboard.pipeline.validate import validate_attributes

SPECS = [
    AttributeSpec(key="size", type="enum", allowed=["S", "M", "L"], multi=True),
    AttributeSpec(key="gender", type="enum", allowed=["men", "women"]),
    AttributeSpec(key="pack_count", type="int", min=1, max=10),
    AttributeSpec(key="capacity_oz", type="float", min=1, max=100),
    AttributeSpec(key="material", type="string", max_len=8),
    AttributeSpec(key="sku", type="string", pattern=r"^[A-Z]{3}-\d{3}$"),
    AttributeSpec(key="organic", type="bool"),
]
GOOD = {
    "size": ["s", "L"],
    "gender": "Men",
    "pack_count": "3",
    "capacity_oz": 12,
    "material": "cotton",
    "sku": "ABC-123",
    "organic": "yes",
}


def codes(result):
    return sorted(c.code for c in result[1])


def test_happy_path_coerces_values():
    out, cs = validate_attributes(SPECS, GOOD)
    assert cs == []
    assert out == {
        "size": ["S", "L"],
        "gender": "men",
        "pack_count": 3,
        "capacity_oz": 12.0,
        "material": "cotton",
        "sku": "ABC-123",
        "organic": True,
    }


@pytest.mark.parametrize(
    "patch,expected",
    [
        ({"size": None}, ["ATTR_MISSING:size"]),
        ({"gender": []}, ["ATTR_MISSING:gender"]),
        ({"gender": "kids"}, ["ATTR_ENUM_INVALID:gender"]),
        ({"size": ["S", "XXL"]}, ["ATTR_ENUM_INVALID:size"]),
        ({"pack_count": 0}, ["ATTR_RANGE:pack_count"]),
        ({"pack_count": "lots"}, ["ATTR_RANGE:pack_count"]),
        ({"capacity_oz": 100.5}, ["ATTR_RANGE:capacity_oz"]),
        ({"material": "polyester!"}, ["ATTR_TOO_LONG:material"]),
        ({"gender": ["men", "women"]}, ["ATTR_MULTI_NOT_ALLOWED:gender"]),
        ({"sku": "abc-123"}, ["ATTR_PATTERN:sku"]),
        ({"organic": "maybe"}, ["ATTR_ENUM_INVALID:organic"]),
    ],
)
def test_each_branch(patch, expected):
    attrs = {**GOOD, **patch}
    attrs = {k: v for k, v in attrs.items() if v is not None}
    result = validate_attributes(SPECS, attrs)
    assert codes(result) == expected
    assert all(c.severity == "blocking" for c in result[1])


def test_single_element_list_collapses_for_scalar():
    out, cs = validate_attributes(SPECS, {**GOOD, "gender": ["women"]})
    assert cs == [] and out["gender"] == "women"


def test_scalar_wraps_for_multi():
    out, cs = validate_attributes(SPECS, {**GOOD, "size": "m"})
    assert cs == [] and out["size"] == ["M"]


def test_missing_key_is_reported_not_raised():
    _, cs = validate_attributes(SPECS, {})
    assert len(cs) == len(SPECS) and all(c.code.startswith("ATTR_MISSING:") for c in cs)


def test_reconcile_det_wins_and_defaults():
    specs = [SPECS[1], SPECS[2], SPECS[4]]
    attrs, extra, cs = merge_attributes(
        specs, {"gender": "men", "color": ["red"]}, {"gender": "women", "material": "wool"}
    )
    assert attrs == {"gender": "men", "material": "wool", "pack_count": 1}
    assert extra == {"color": ["red"]}
    names = {c.code for c in cs}
    assert {"ATTR_CONFLICT:gender", "ATTR_EXTRA:color", "ATTR_DEFAULTED:pack_count"} == names
    assert all(c.severity != "blocking" for c in cs)
    assert [s.key for s in missing_specs(specs, {"gender": "men"})] == ["material"]


def test_validation_modules_do_not_import_llm():
    from tests.helpers import assert_modules_do_not_load

    assert_modules_do_not_load(
        ["catalog_onboard.pipeline.validate", "catalog_onboard.pipeline.comply", "catalog_onboard.pipeline.reconcile"],
        forbidden_prefix="catalog_onboard.pipeline.llm",
    )
