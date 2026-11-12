"""Online ranker: parity with offline scoring, fallback injection, truncation (spec §13.4).
Latency under load lives in tests/load/."""

import numpy as np
import pandas as pd
import pytest

from shopfeed import config
from shopfeed.features.ranking import build_features
from shopfeed.serving.ranker import Ranker, fallback_order
from shopfeed.serving.rules import RuleFlags
from shopfeed.training.train_lambdamart import fit, write_artifacts

SMOKE = {"min_data_in_leaf": 1, "min_data_in_bin": 1, "num_leaves": 4, "bagging_fraction": 1.0,
         "feature_fraction": 1.0, "num_threads": 1}
NO_RULES = RuleFlags(False, False, False, False)
# Parity tests must not depend on runner speed; timeout behaviour has its own test.
PARITY_TIMEOUT_S = 10.0


@pytest.fixture(scope="module")
def artefact(fixture_rows, fixture_store, tmp_path_factory):
    X, names = build_features(fixture_rows, fixture_store)
    booster = fit(fixture_rows, X, fixture_rows, X, params=SMOKE, num_boost_round=10,
                  early_stopping=10)
    out = tmp_path_factory.mktemp("ranker")
    write_artifacts(out, booster, names, {}, {"git_sha": "abc123"})
    return out, booster, X


def _req(rows, qid=1):
    return rows[rows["qid"] == qid].reset_index(drop=True)


def test_serving_scores_match_offline(fixture_rows, fixture_store, artefact):
    out, booster, X = artefact
    ranker = Ranker.load(out, fixture_store, flags=NO_RULES, timeout_s=PARITY_TIMEOUT_S)
    req = _req(fixture_rows)
    res = ranker.rank(req)
    assert not res.fallback, res.reason
    offline = booster.predict(X[fixture_rows["qid"] == 1])
    expected = req.assign(s=offline).sort_values("s", ascending=False, kind="stable")
    assert res.item_ids == expected["item_id"].tolist()
    np.testing.assert_allclose(res.scores, expected["s"].to_numpy())
    assert ranker.model_version == "abc123"


def test_request_log_features_match_recomputed(fixture_rows, fixture_store, artefact):
    """Train-serve skew check: logged features equal an offline recomputation."""
    out, _, X = artefact
    ranker = Ranker.load(out, fixture_store, flags=NO_RULES, timeout_s=PARITY_TIMEOUT_S)
    ranker.rank(_req(fixture_rows, 2))
    logged = np.array(ranker.request_log[-1]["features"])
    recomputed, _ = build_features(_req(fixture_rows, 2), fixture_store)
    np.testing.assert_allclose(logged, recomputed[ranker.feature_names].to_numpy(),
                               equal_nan=True)


@pytest.mark.parametrize("break_what", ["model", "store"])
def test_failures_fall_back_to_bm25_order(fixture_rows, fixture_store, artefact, break_what):
    out, _, _ = artefact
    ranker = Ranker.load(out, fixture_store, timeout_s=PARITY_TIMEOUT_S)
    if break_what == "model":
        ranker.score = lambda X: (_ for _ in ()).throw(RuntimeError("model down"))
    else:
        ranker.store = None  # feature store outage
    req = _req(fixture_rows)
    res = ranker.rank(req)
    assert res.fallback and res.item_ids == fallback_order(req)
    assert res.item_ids[0] == "p_301"  # highest bm25 (14.8)
    assert ranker.counters["fallback"] == 1


def test_timeout_falls_back(fixture_rows, fixture_store, artefact):
    out, _, _ = artefact
    ranker = Ranker.load(out, fixture_store, timeout_s=0.0)
    res = ranker.rank(_req(fixture_rows))
    assert res.fallback and res.reason == "timeout"


def test_fallback_with_missing_bm25_keeps_retrieval_order(fixture_rows):
    req = _req(fixture_rows).assign(bm25=np.nan)
    assert fallback_order(req) == req["item_id"].tolist()


def test_more_than_max_candidates_is_truncated(fixture_rows, fixture_store, artefact):
    out, _, _ = artefact
    ranker = Ranker.load(out, fixture_store, flags=NO_RULES, timeout_s=PARITY_TIMEOUT_S)
    base = _req(fixture_rows)
    n = config.MAX_CANDIDATES + 50
    big = pd.concat([base] * (n // len(base) + 1), ignore_index=True).head(n)
    big["item_id"] = [f"p_{900000 + i}" for i in range(n)]  # unique, unknown to the store
    res = ranker.rank(big)
    assert not res.fallback, res.reason
    assert len(res.item_ids) == config.MAX_CANDIDATES
    assert set(res.item_ids) == set(big["item_id"].head(config.MAX_CANDIDATES))
