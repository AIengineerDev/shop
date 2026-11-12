import pandas as pd
import pytest

from shopfeed import config
from shopfeed.schema import COLUMNS, SchemaError, read_csv, validate


def test_fixture_matches_contract():
    df = read_csv(config.FIXTURE_CSV)
    validate(df)
    assert len(df) == 10
    assert df["qid"].nunique() == 2


def test_header_is_spec_v2():
    assert COLUMNS[-4:] == ["impression_ts", "wishlisted", "seller_id", "category"]
    assert "dwell_s" not in COLUMNS


def test_validate_rejects_missing_column():
    df = read_csv(config.FIXTURE_CSV).drop(columns=["category"])
    with pytest.raises(SchemaError, match="missing"):
        validate(df)


def test_validate_rejects_null_label():
    df = read_csv(config.FIXTURE_CSV)
    df["clicked"] = df["clicked"].astype("float64")
    df.loc[0, "clicked"] = float("nan")
    df["clicked"] = df["clicked"].astype("Int64")
    with pytest.raises(SchemaError):
        validate(df)


def test_validate_rejects_wrong_order():
    df = read_csv(config.FIXTURE_CSV)
    with pytest.raises(SchemaError):
        validate(df[list(reversed(df.columns))])


def test_read_csv_keeps_empty_query_as_string(tmp_path):
    df = read_csv(config.FIXTURE_CSV)
    df.loc[:, "query_text"] = ""
    path = tmp_path / "x.csv"
    df.to_csv(path, index=False)
    out = read_csv(path)
    assert (out["query_text"] == "").all()
    assert isinstance(out, pd.DataFrame)
