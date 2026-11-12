"""Spec §6.1 baselines, scored through the M3 harness.

0  logged     : the logged order (randomised in the current log), score = -displayed_pos
1  bm25       : N/A, bm25 is not in the feed log (reported as skipped)
2  ctr        : hist_ctr_7d
2a bestseller : item_purchases_30d (the bar the popularity prior must clear, §5.5)
2b semantic   : semantic_sim (user-item proxy); added because it is the strongest
                single feature on this data
3  pointwise  : LightGBM binary classifier on `clicked`
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from shopfeed import config

SKIPPED = {"bm25": "bm25 is NaN in the feed log (no query text); BM25 sort not evaluable"}


def heuristic_scores(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "logged": -rows["displayed_pos"].astype(float),
        "ctr": rows["hist_ctr_7d"].fillna(-1.0),
        "bestseller": rows["item_purchases_30d"].astype(float),
        "semantic": rows["semantic_sim"].fillna(-2.0),
    }, index=rows.index)


def train_pointwise(X_train, y_train, X_val, y_val, seed: int = config.SEED) -> lgb.Booster:
    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31,
              "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "seed": seed, "deterministic": True, "num_threads": 1,
              "verbose": -1}
    train = lgb.Dataset(X_train, y_train)
    val = lgb.Dataset(X_val, y_val, reference=train)
    return lgb.train(params, train, num_boost_round=1000, valid_sets=[val],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def all_scores(rows_test, X_test, booster: lgb.Booster | None) -> pd.DataFrame:
    s = heuristic_scores(rows_test)
    if booster is not None:
        s["pointwise"] = booster.predict(X_test, num_iteration=booster.best_iteration)
    return s.astype(np.float64)
