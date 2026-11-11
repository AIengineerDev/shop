from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from catalog_onboard.models import ReasonCode, ValidationResult
from catalog_onboard.pipeline.comply import check_compliance, find_prohibited, fingerprint
from catalog_onboard.pipeline.route import route


@pytest.mark.parametrize(
    "text,kw,hit",
    [
        ("a sharp knife", "knife", True),
        ("knifeless design", "knife", False),
        ("Gunnar brand", "gun", False),
        ("toy GUN here", "gun", True),
        ("bear  spray holster", "bear spray", True),
        ("bearspray", "bear spray", False),
        ("(replica)", "replica", True),
    ],
)
def test_word_boundary_matching(text, kw, hit):
    assert (find_prohibited(text, [kw]) == [kw]) is hit


def run(**kw):
    base = dict(
        clean_title="Good Title",
        scan_text="Good Title bullet",
        category_prohibited=(),
        global_prohibited=(),
        price_min=Decimal("1"),
        duplicate_exists=lambda: False,
    )
    base.update(kw)
    return [c.code for c in check_compliance(**base)]


def test_clean_listing_has_no_codes():
    assert run() == []


def test_ac_3_2_all_codes_collected():
    codes = run(
        clean_title="ab",
        scan_text="ab replica switchblade",
        category_prohibited=("switchblade",),
        global_prohibited=("replica",),
        price_min=Decimal("0"),
        duplicate_exists=lambda: True,
    )
    assert codes == [
        "PROHIBITED_KEYWORD:switchblade",
        "PROHIBITED_GLOBAL:replica",
        "PRICE_INVALID",
        "TITLE_TOO_SHORT",
        "DUPLICATE_LISTING",
    ]


def test_price_missing_is_invalid():
    assert run(price_min=None) == ["PRICE_INVALID"]


def test_fingerprint_stable_and_case_insensitive():
    a = fingerprint("s", "Men Shorts", {"size": ["L"], "color": ["navy"]})
    b = fingerprint("s", "men shorts", {"color": ["navy"], "size": ["L"]})
    assert a == b and a != fingerprint("other", "men shorts", {"color": ["navy"], "size": ["L"]})


severity = st.sampled_from(["blocking", "warning", "info"])
code = st.builds(
    ReasonCode,
    code=st.text(min_size=1, max_size=12),
    severity=severity,
    stage=st.sampled_from(["clean", "validate", "comply"]),
    detail=st.none(),
)


@given(st.lists(code, max_size=8))
def test_ac_3_1_routing_invariant(codes):
    result = ValidationResult(reason_codes=codes)
    has_blocking = any(c.severity == "blocking" for c in codes)
    assert route(result) == ("quarantined" if has_blocking else "draft")
    assert result.ok is not has_blocking
