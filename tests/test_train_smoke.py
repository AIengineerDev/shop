"""LambdaMART smoke + reproducibility on the 10-row fixture (spec §13.3)."""

import json

import numpy as np
import pandas as pd
import pytest

from shopfeed import config
from shopfeed.features.history import HistoryStore
from shopfeed.features.ranking import EXCLUDED, build_features
from shopfeed.schema import read_csv
from shopfeed.training.dataset import EVAL_COLUMNS
from shopfeed.training.train_lambdamart import (
    evaluate_test,
    fit,
    importance,
    predict,
    write_artifacts,
)

SMOKE = {"min_data_in_leaf": 1, "min_data_in_bin": 1, "num_leaves": 4, "bagging_fraction": 1.0,
         "feature_fraction": 1.0, "num_threads": 1}
ARTEFACTS = {"model.txt", "features.json", "propensities.json", "metrics.json", "meta.json",
             "importance.csv"}


@pytest.fixture(scope="module")
def fixture_xy():
    rows = read_csv(config.FIXTURE_CSV)
    empty = pd.DataFrame(columns=["event_id", "user_id", "product_id", "merchant_id", "type",
                                  "ts", "impression_id", "value"]).astype({"ts": float})
    products = pd.DataFrame({"product_id": rows["item_id"], "merchant_id": rows["seller_id"],
                             "category": rows["category"],
                             "published_at": rows["impression_ts"] - 86400})
    merchants = pd.DataFrame({"merchant_id": rows["seller_id"].unique(), "return_rate": 0.1,
                              "fulfilment_days": 2.0})
    store = HistoryStore.from_sources(
        pd.DataFrame({"product_id": rows["item_id"], "ts": rows["impression_ts"] - 1}),
        empty, products, merchants)
    X, names = build_features(rows, store)
    return rows, X, names


def _fit(rows, X, rounds=5):
    return fit(rows, X, rows, X, params=SMOKE, num_boost_round=rounds, early_stopping=rounds)


def test_smoke_writes_all_artefacts(fixture_xy, tmp_path):
    rows, X, names = fixture_xy
    booster = _fit(rows, X)
    rows_eval = pd.concat([rows, X[EVAL_COLUMNS]], axis=1)
    res = evaluate_test(rows_eval, predict(booster, X), n_boot=20)
    write_artifacts(tmp_path, booster, names, {"gates": res["gates"]}, {"seed": config.SEED})
    assert {p.name for p in tmp_path.iterdir()} == ARTEFACTS
    spec = json.loads((tmp_path / "features.json").read_text())
    assert spec["features"] == names and not set(spec["features"]) & EXCLUDED


def test_reproducible(fixture_xy):
    rows, X, _ = fixture_xy
    a, b = predict(_fit(rows, X), X), predict(_fit(rows, X), X)
    np.testing.assert_array_equal(a, b)


def test_learns_fixture_order(fixture_xy):
    rows, X, _ = fixture_xy
    s = predict(_fit(rows, X, rounds=20), X)
    for _, g in rows.assign(s=s).groupby("qid"):
        assert g.loc[g["s"].idxmax(), "relevance_grade"] == 4


def test_importance_shares_sum_to_one(fixture_xy):
    rows, X, _ = fixture_xy
    imp = importance(_fit(rows, X, rounds=20))
    assert imp["gain_share"].sum() == pytest.approx(1.0)
