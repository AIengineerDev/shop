"""Integration: real module boundaries on a synthetic log (build -> clean -> checks ->
features -> split -> train -> artefact -> serving). Catches contract breaks between
milestones that unit tests with hand-made frames cannot."""

import json

import numpy as np
import pandas as pd
import pytest

from shopfeed import config
from shopfeed.data.build import Sources
from shopfeed.data.checks import run_all
from shopfeed.features.ranking import EXCLUDED, build_features
from shopfeed.schema import COLUMNS, read_csv
from shopfeed.serving.cache import RankedListCache
from shopfeed.serving.pagination import Paginator
from shopfeed.serving.ranker import Ranker
from shopfeed.serving.rules import RuleFlags

pytestmark = pytest.mark.integration


def test_build_clean_checks_contract(synth_root):
    assert synth_root["clean_report"]["failed_rules"] == []
    assert synth_root["clean_report"]["rows_out"] == synth_root["build_report"]["rows_out"]
    df = read_csv(synth_root["data"] / "ranking_items.csv")
    assert list(df.columns) == COLUMNS
    results = run_all(df, Sources.load(synth_root["data"]), profile="ci")
    failed = [(r.name, r.detail) for r in results if not r.passed]
    assert not failed


def test_dataset_split_and_features(synth_dataset):
    ds = synth_dataset
    assert len(ds.X) == len(ds.rows)
    assert not set(ds.feature_names) & EXCLUDED
    parts = ds.rows["partition"].value_counts()
    assert set(parts.index) == {"train", "val", "test"}
    assert ds.split.sessions["train"] > ds.split.sessions["val"] > 0
    t = ds.rows.groupby("partition")["impression_ts"].agg(["min", "max"])
    assert t.loc["train", "max"] < t.loc["val", "min"] <= t.loc["val", "max"] < t.loc["test", "min"]


def test_training_writes_artefacts_and_learns(synth_model):
    out = synth_model["dir"]
    for f in ("model.txt", "features.json", "propensities.json", "metrics.json", "meta.json",
              "importance.csv"):
        assert (out / f).exists(), f
    m = json.loads((out / "metrics.json").read_text())
    assert m["gates"]["ndcg@10_rel_vs_logged"]["pass"]
    p = json.loads((out / "propensities.json").read_text())["propensity"]
    assert p[0] == 1.0 and p[-1] < 0.9  # synthetic examination decay is recovered


def _request(ds, n_groups=10):
    te, _ = ds.part("test")
    qids = te["qid"].unique()[:n_groups]
    req = te[te["qid"].isin(qids)].copy()
    req["qid"] = int(qids[0])
    return req.drop_duplicates("item_id").reset_index(drop=True)


def test_serving_matches_offline_scores(synth_dataset, synth_model):
    ranker = Ranker.load(synth_model["dir"], synth_dataset.store,
                         flags=RuleFlags(False, False, False, False), timeout_s=10)
    req = _request(synth_dataset)
    res = ranker.rank(req)
    assert not res.fallback, res.reason
    X, _ = build_features(req, synth_dataset.store)
    offline = synth_model["booster"].predict(X[ranker.feature_names].to_numpy(),
                                             num_iteration=synth_model["booster"].best_iteration)
    expected = req.assign(s=offline).sort_values("s", ascending=False, kind="stable")
    assert res.item_ids == expected["item_id"].tolist()
    np.testing.assert_allclose(res.scores, expected["s"].to_numpy())


def test_rules_preserve_candidate_set(synth_dataset, synth_model):
    ranker = Ranker.load(synth_model["dir"], synth_dataset.store, timeout_s=10)
    req = _request(synth_dataset)
    res = ranker.rank(req)
    assert sorted(res.item_ids) == sorted(req["item_id"])
    assert len(res.item_ids) == len(set(res.item_ids))


def test_request_log_has_no_train_serve_skew(synth_dataset, synth_model):
    ranker = Ranker.load(synth_model["dir"], synth_dataset.store, timeout_s=10)
    req = _request(synth_dataset, 3)
    ranker.rank(req)
    logged = np.array(ranker.request_log[-1]["features"])
    X, _ = build_features(req, synth_dataset.store)
    np.testing.assert_allclose(logged, X[ranker.feature_names].to_numpy(), equal_nan=True)


def test_ranker_with_pagination_and_cache(synth_dataset, synth_model):
    ranker = Ranker.load(synth_model["dir"], synth_dataset.store, timeout_s=10)
    req = _request(synth_dataset)
    calls = []

    def rank_fn(qid):
        calls.append(qid)
        return ranker.rank(req).item_ids

    p = Paginator(rank_fn, RankedListCache(), page_size=20)
    page = p.first_page(int(req["qid"].iloc[0]))
    served = list(page.item_ids)
    while page.next_cursor:
        page = p.next_page(page.next_cursor)
        served += page.item_ids
    assert len(served) == len(set(served)) == len(req)
    assert len(calls) == 1 and p.cache.hit_rate() == 1.0


def test_duplicate_candidates_are_dropped(synth_dataset, synth_model):
    ranker = Ranker.load(synth_model["dir"], synth_dataset.store, timeout_s=10)
    req = _request(synth_dataset, 2)
    res = ranker.rank(pd.concat([req, req.iloc[:3]], ignore_index=True))
    assert len(res.item_ids) == len(set(res.item_ids)) == len(req)
    assert ranker.counters["duplicate_candidates"] == 3


def test_store_outage_degrades_to_fallback(synth_dataset, synth_model):
    ranker = Ranker.load(synth_model["dir"], synth_dataset.store, timeout_s=10)
    ranker.store = None
    req = _request(synth_dataset, 2)
    res = ranker.rank(req)
    assert res.fallback and sorted(res.item_ids) == sorted(req["item_id"])


def test_full_dataset_features_identical_in_batch_and_per_request(synth_dataset):
    """Within-group features computed on the whole test split equal per-request ones."""
    te, X_te = synth_dataset.part("test")
    one = te[te["qid"] == te["qid"].iloc[0]].reset_index(drop=True)
    X_one, _ = build_features(one, synth_dataset.store)
    batch = X_te[(te["qid"] == one["qid"].iloc[0]).to_numpy()].reset_index(drop=True)
    pd.testing.assert_frame_equal(batch, X_one, check_exact=False, rtol=1e-12)


def test_config_paths_default_to_repo():
    assert config.DATA_DIR.name == "data" or "SHOPFEED_DATA_DIR" in __import__("os").environ


def test_readiness_check_catches_ctr_leakage(synth_root):
    """hist_ctr_7d recomputed from raw events must match; a tampered column must fail."""
    df = read_csv(synth_root["data"] / "ranking_items.csv")
    df["hist_ctr_7d"] = df["hist_ctr_7d"].fillna(0.0) + 0.05
    results = {r.name: r for r in run_all(df, Sources.load(synth_root["data"]), profile="ci")}
    assert not results["hist_ctr_point_in_time"].passed
    assert "mismatches=" in results["hist_ctr_point_in_time"].detail


def test_sweep_selects_by_validation(synth_dataset):
    from shopfeed.training.sweep import select, sweep

    res = sweep(synth_dataset, n_trials=2, gains=(10, 40), num_boost_round=50,
                early_stopping=10)
    assert len(res) == 4 and set(res["purchase_gain"]) == {10, 40}
    assert res[["val_purchase_mrr", "val_ndcg@10"]].notna().all().all()
    best = select(res)
    assert best["val_purchase_mrr"] == res["val_purchase_mrr"].max()


def test_pointwise_baseline_trains_and_scores(synth_dataset):
    from shopfeed.training.baselines import all_scores, train_pointwise

    tr_rows, tr_X = synth_dataset.part("train")
    va_rows, va_X = synth_dataset.part("val")
    te_rows, te_X = synth_dataset.part("test")
    booster = train_pointwise(tr_X, tr_rows["clicked"], va_X, va_rows["clicked"])
    scores = all_scores(te_rows, te_X, booster)
    assert set(scores.columns) == {"logged", "ctr", "bestseller", "semantic", "pointwise"}
    assert scores.notna().all().all() and len(scores) == len(te_rows)
    assert scores["pointwise"].between(0, 1).all()
