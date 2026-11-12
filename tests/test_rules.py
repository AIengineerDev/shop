"""§5.5 post-ranking rules on a crafted 12-item group (spec §13.1)."""

import numpy as np
import pandas as pd

from shopfeed.serving.rules import RuleFlags, apply_rules

OFF = RuleFlags(relevance_floor=False, unpopular_demotion=False, new_item_exploration=False,
                bestseller_boost=False)


def _group(**overrides):
    n = 12
    df = pd.DataFrame({
        "qid": 0,
        "item_id": [f"p_{i}" for i in range(n)],
        "score": np.linspace(1.0, 0.0, n),  # p_0 best ... p_11 worst
        "semantic_sim": np.linspace(0.9, 0.1, n),
        "is_unpopular": 0,
        "item_age_days": 100.0,
        "item_impressions_30d": 1000,
        "item_bestseller_rank_category": np.arange(2, n + 2),
        "is_category_top20": 1,
    })
    for k, v in overrides.items():
        df[k] = v
    return df


def _ids(df):
    return list(df["item_id"])


def test_all_flags_off_is_score_order():
    out = apply_rules(_group(), OFF)
    assert _ids(out) == [f"p_{i}" for i in range(12)]
    assert list(out["rank"]) == list(range(1, 13))


def test_relevance_floor_keeps_low_similarity_out_of_top3():
    sim = np.linspace(0.9, 0.1, 12)
    sim[0] = 0.0  # top-scored item is below the group median similarity
    out = apply_rules(_group(semantic_sim=sim), RuleFlags(unpopular_demotion=False,
                                                          new_item_exploration=False))
    assert _ids(out)[:4] == ["p_1", "p_2", "p_3", "p_0"]


def test_unpopular_demoted_below_5_but_kept():
    unpop = np.zeros(12, int)
    unpop[[0, 1]] = 1
    out = apply_rules(_group(is_unpopular=unpop), RuleFlags(relevance_floor=False,
                                                            new_item_exploration=False))
    assert _ids(out)[:5] == ["p_2", "p_3", "p_4", "p_5", "p_6"]
    assert _ids(out)[5:7] == ["p_0", "p_1"] and len(out) == 12


def test_new_item_exempt_from_demotion():
    unpop = np.zeros(12, int)
    unpop[0] = 1
    age = np.full(12, 100.0)
    age[0] = 3.0
    imps = np.full(12, 1000)
    imps[0] = 10
    out = apply_rules(_group(is_unpopular=unpop, item_age_days=age, item_impressions_30d=imps),
                      RuleFlags(relevance_floor=False, new_item_exploration=False))
    assert _ids(out)[0] == "p_0"


def test_new_item_exploration_slot_round_robin():
    age = np.full(12, 100.0)
    age[11] = 2.0
    imps = np.full(12, 1000)
    imps[11] = 0
    for qid in (0, 3):
        out = apply_rules(_group(item_age_days=age, item_impressions_30d=imps, qid=qid),
                          RuleFlags(relevance_floor=False, unpopular_demotion=False))
        assert _ids(out).index("p_11") == 5 + qid % 5
        assert sorted(_ids(out)) == sorted(f"p_{i}" for i in range(12))


def test_bestseller_boost_off_by_default_and_moves_to_10_when_on():
    rank = np.arange(2, 14)
    rank[11] = 1  # worst-scored item is the category #1 seller
    g = _group(item_bestseller_rank_category=rank)
    assert _ids(apply_rules(g, OFF))[-1] == "p_11"
    on = RuleFlags(relevance_floor=False, unpopular_demotion=False, new_item_exploration=False,
                   bestseller_boost=True)
    assert _ids(apply_rules(g, on))[9] == "p_11"
