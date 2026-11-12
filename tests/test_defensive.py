"""Defensive branches: error paths and edge cases that normal data never reaches."""

import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from shopfeed import config
from shopfeed.data import clean as C
from shopfeed.eval.propensity import estimate_from_randomised
from shopfeed.features.history import _count, _key_groups
from shopfeed.schema import SchemaError, read_csv, validate
from shopfeed.serving.cache import TTLCache
from shopfeed.training import train_lambdamart as T


def test_git_sha_falls_back_when_git_unavailable(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "check_output", boom)
    assert T.git_sha() == "unknown"


def test_git_sha_in_repo():
    sha = T.git_sha()
    assert sha == "unknown" or len(sha) == 40


def test_group_sizes_requires_sorted_qid():
    with pytest.raises(ValueError, match="sorted by qid"):
        T.group_sizes(pd.DataFrame({"qid": [2, 1, 2]}))
    assert list(T.group_sizes(pd.DataFrame({"qid": [1, 1, 2]}))) == [2, 1]


@pytest.mark.parametrize("value,expected", [
    (np.int64(3), 3), (np.float64(1.5), 1.5), (np.float64("nan"), None), (np.bool_(True), True),
])
def test_json_default_encodes_numpy(value, expected):
    assert T._json_default(value) == expected


def test_json_default_rejects_unknown_types():
    with pytest.raises(TypeError):
        T._json_default(object())


def test_clean_rejects_wrong_header():
    raw = C.read_raw(config.FIXTURE_CSV).drop(columns=["category"])
    with pytest.raises(C.CleanError, match="header"):
        C.clean(raw)


def test_write_report_creates_parent_dirs(tmp_path):
    path = tmp_path / "a" / "b" / "clean_report.json"
    C.write_report({"rules": {}}, path)
    assert json.loads(path.read_text()) == {"rules": {}}


def test_validate_rejects_null_group_key():
    df = read_csv(config.FIXTURE_CSV)
    df.loc[0, "session_id"] = pd.NA  # dtype stays "string", so only the null check fires
    with pytest.raises(SchemaError, match="nulls"):
        validate(df)


def test_key_grouping_on_empty_and_null_keys():
    assert list(_key_groups([])) == []
    assert _count({"a": np.array([1.0])}, [], np.array([]), None).shape == (0,)
    groups = dict(_key_groups(["a", None, "a"]))
    assert list(groups) == ["a"] and list(groups["a"]) == [0, 2]


def test_propensity_max_pos_truncates():
    rows = pd.DataFrame({"displayed_pos": [1, 1, 2, 2, 3, 3], "clicked": [1, 1, 1, 0, 1, 0]})
    assert estimate_from_randomised(rows, max_pos=2)["positions"] == [1, 2]


def test_ttl_cache_keys():
    c = TTLCache(10)
    c.set("a", 1)
    c.set("b", 2)
    assert sorted(c.keys()) == ["a", "b"]
