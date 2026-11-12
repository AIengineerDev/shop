"""Metric implementations against hand-computed values (spec §13.1)."""

import numpy as np
import pandas as pd
import pytest

from shopfeed import config
from shopfeed.eval import metrics as M
from shopfeed.eval.offline import bootstrap_ci, evaluate, paired_delta, write_html
from shopfeed.schema import read_csv

# grades in score order: 3, 2, 0, 1 ; ideal order: 3, 2, 1, 0
# purchase-dominant gains {0,1,2,5,20}: DCG = 5 + 2/log2(3) + 1/log2(5)
#   = 5 + 1.2618595071429148 + 0.43067655807339306 = 6.692536065216308
# IDCG = 5 + 2/log2(3) + 1/log2(4) = 6.7618595071429155
NDCG_PD = 6.692536065216308 / 6.7618595071429155
# exponential gains {0,1,3,7,15}: DCG = 7 + 3/log2(3) + 1/log2(5) = 9.323465818787767
# IDCG = 7 + 3/log2(3) + 0.5 = 9.392789260714373
NDCG_EXP = 9.323465818787767 / 9.392789260714373


def test_ndcg_hand_computed():
    grades, scores = [3, 2, 0, 1], [4, 3, 2, 1]
    assert M.ndcg_at_k(grades, scores, 10) == pytest.approx(NDCG_PD, abs=1e-9)
    assert M.ndcg_at_k(grades, scores, 10, config.GAIN_EXPONENTIAL) == pytest.approx(
        NDCG_EXP, abs=1e-9)


def test_ndcg_cutoff_and_edge_cases():
    assert M.ndcg_at_k([0, 4], [2, 1], 1) == 0.0
    assert M.ndcg_at_k([4, 0], [2, 1], 1) == 1.0
    assert np.isnan(M.ndcg_at_k([0, 0], [1, 2], 10))


def test_ties_are_stable():
    assert M.ndcg_at_k([0, 4], [1, 1], 1) == 0.0  # first row wins the tie


def test_purchase_metrics():
    purchased, scores = [0, 0, 1, 0], [4, 3, 2, 1]
    assert M.purchase_reciprocal_rank(purchased, scores) == pytest.approx(1 / 3)
    assert M.purchase_hits_at_k(purchased, scores, 2) == (0, 1)
    assert M.purchase_hits_at_k(purchased, scores, 3) == (1, 1)
    assert np.isnan(M.purchase_reciprocal_rank([0, 0], [1, 2]))


def test_average_precision():
    # relevant (grade >= 2) at ranks 1 and 3 -> (1/1 + 2/3) / 2
    assert M.average_precision([2, 0, 4, 1], [4, 3, 2, 1]) == pytest.approx((1 + 2 / 3) / 2)


def _fixture():
    df = read_csv(config.FIXTURE_CSV)
    df["logged"] = -df["displayed_pos"].astype(float)
    df["reverse"] = df["displayed_pos"].astype(float)
    return df


def test_per_query_and_summary_on_fixture():
    pq = M.per_query(_fixture(), "logged")
    s = M.summarise(pq)
    assert s["purchase_mrr"] == 1.0 and s["purchase@1"] == 1.0  # both purchases at pos 1
    assert s["ndcg@10"] == pytest.approx(
        np.mean([M.ndcg_at_k([4, 3, 0, 2, 0], [5, 4, 3, 2, 1]),
                 M.ndcg_at_k([4, 1, 0, 0, 0], [5, 4, 3, 2, 1])]))
    assert s["mean_price@10"] == pytest.approx(_fixture().groupby("qid")["price"].mean().mean())


def test_bootstrap_ci_brackets_point():
    pq = M.per_query(_fixture(), "reverse")
    point, lo, hi = bootstrap_ci(pq, "ndcg@10", n=200)
    assert lo <= point <= hi


def test_paired_delta_sign():
    df = _fixture()
    d = paired_delta(M.per_query(df, "reverse"), M.per_query(df, "logged"), "ndcg@10", n=200)
    assert d["delta"] < 0 and d["delta_hi"] < 0
    r = paired_delta(M.per_query(df, "logged"), M.per_query(df, "logged"), "purchase@10", n=50)
    assert r["delta"] == 0.0


def test_evaluate_and_html(tmp_path):
    df = _fixture()
    df["is_category_top20"] = (df["displayed_pos"] <= 2).astype(int)
    res = evaluate(df, ["logged", "reverse"], reference="logged", n_boot=50)
    assert set(res) == {"overall", "deltas", "slices"}
    assert {"ndcg@10", "purchase_mrr", "purchase@10"} <= set(res["overall"]["metric"])
    assert "purchased_item_popularity" in set(res["slices"]["slice"])
    path = write_html(res, tmp_path / "r.html", "test")
    assert "Paired deltas" in path.read_text()


def test_longtail_flag():
    df = pd.DataFrame({"item_id": list("abcde"), "item_purchases_30d": [0, 0, 1, 2, 9]})
    assert list(M.add_longtail_flag(df)["is_longtail"]) == [1, 1, 1, 1, 0]
