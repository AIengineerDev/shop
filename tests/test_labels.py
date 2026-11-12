"""Grade ladder (spec §4.2, §13.1) and §4.3 consistency."""

import pandas as pd
import pytest

from shopfeed import config
from shopfeed.labels import backfill_clicked, consistency_violations, grade
from shopfeed.schema import read_csv


@pytest.mark.parametrize("clicked,wish,atc,purch,hidden,expected", [
    (0, 0, 0, 0, 0, 0),  # no click
    (1, 0, 0, 0, 0, 1),  # click
    (1, 1, 0, 0, 0, 2),  # click + wishlist
    (0, 1, 0, 0, 0, 2),  # wishlist, no click
    (1, 0, 1, 0, 0, 3),  # ATC
    (1, 1, 1, 0, 0, 3),  # ATC + wishlist
    (1, 0, 0, 1, 0, 4),  # purchase
    (1, 0, 0, 0, 1, 0),  # click then hide
    (1, 1, 0, 0, 1, 2),  # wishlist outranks hide
    (1, 0, 0, 1, 1, 4),  # purchase outranks hide
])
def test_grade_ladder(clicked, wish, atc, purch, hidden, expected):
    df = pd.DataFrame([{"clicked": clicked, "wishlisted": wish, "added_to_cart": atc,
                        "purchased": purch}])
    assert grade(df, pd.Series([hidden])).item() == expected


def test_backfill_clicked():
    df = pd.DataFrame({"clicked": [0, 0, 0, 0], "wishlisted": [1, 0, 0, 0],
                       "added_to_cart": [0, 1, 0, 0], "purchased": [0, 0, 1, 0]})
    assert list(backfill_clicked(df)) == [1, 1, 1, 0]


def test_fixture_is_consistent():
    df = read_csv(config.FIXTURE_CSV)
    assert consistency_violations(df).sum() == 0
    assert (grade(df) == df["relevance_grade"]).all()


@pytest.mark.parametrize("mutation", [
    {"purchased": 1, "relevance_grade": 3},
    {"added_to_cart": 1, "relevance_grade": 2},
    {"wishlisted": 1, "relevance_grade": 1},
    {"clicked": 0, "relevance_grade": 1},
])
def test_consistency_violations_detected(mutation):
    df = pd.DataFrame([{"clicked": 1, "wishlisted": 0, "added_to_cart": 0, "purchased": 0,
                        "relevance_grade": 1, **mutation}])
    assert consistency_violations(df).item()
