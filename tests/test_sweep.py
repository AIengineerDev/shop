import pandas as pd

from shopfeed.training.sweep import SPACE, gain_table, sample_configs, select


def test_sample_configs_deterministic_and_in_space():
    a, b = sample_configs(30, seed=1), sample_configs(30, seed=1)
    assert a == b and len(a) == 30
    assert all(cfg[k] in SPACE[k] for cfg in a for k in SPACE)


def test_gain_table_only_changes_purchase():
    assert gain_table(40) == [0, 1, 2, 5, 40]


def test_select_by_mrr_then_ndcg():
    r = pd.DataFrame({"val_purchase_mrr": [0.5, 0.6, 0.6], "val_ndcg@10": [0.9, 0.4, 0.5],
                      "trial": [0, 1, 2]})
    assert select(r)["trial"] == 2
