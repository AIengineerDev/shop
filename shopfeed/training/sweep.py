"""Purchase-gain x hyperparameter random search on VALIDATION only (spec §7.3, risk 8).

Selection: highest validation Purchase MRR, tie-break validation NDCG@10. The test split is
not touched here; scripts/sweep.py reads it once for the chosen config via train_lambdamart.run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shopfeed import config
from shopfeed.eval.metrics import per_query, summarise
from shopfeed.training.train_lambdamart import filter_groups, fit, predict

PURCHASE_GAINS = (10, 20, 40)
SPACE = {
    "num_leaves": [7, 15, 31, 63],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "min_data_in_leaf": [50, 100, 200, 500, 1000],
    "feature_fraction": [0.5, 0.7, 0.8, 1.0],
    "lambda_l2": [0.0, 1.0, 5.0, 10.0],
}


def sample_configs(n: int, seed: int = config.SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{k: (v[int(rng.integers(len(v)))]) for k, v in SPACE.items()} for _ in range(n)]


def gain_table(purchase_gain: int) -> list[int]:
    g = list(config.GAIN_PURCHASE_DOMINANT)
    g[4] = purchase_gain
    return g


def sweep(ds, n_trials: int = 30, gains=PURCHASE_GAINS, position: bool = True,
          seed: int = config.SEED, num_boost_round: int = 2000,
          early_stopping: int = 100) -> pd.DataFrame:
    tr_rows, tr_X = filter_groups(*ds.part("train"))
    va_rows_f, va_X_f = filter_groups(*ds.part("val"))
    va_rows, va_X = ds.part("val")
    results = []
    configs = sample_configs(n_trials, seed)
    for gain in gains:
        for i, hp in enumerate(configs):
            params = {**hp, "label_gain": gain_table(gain)}
            booster = fit(tr_rows, tr_X, va_rows_f, va_X_f, params=params, position=position,
                          num_boost_round=num_boost_round, early_stopping=early_stopping)
            s = summarise(per_query(va_rows.assign(score=predict(booster, va_X)), "score"))
            results.append({"purchase_gain": gain, "trial": i, **hp,
                            "best_iteration": booster.best_iteration,
                            "val_purchase_mrr": s["purchase_mrr"], "val_ndcg@10": s["ndcg@10"],
                            "val_purchase@10": s["purchase@10"]})
    return pd.DataFrame(results)


def select(results: pd.DataFrame) -> pd.Series:
    return results.sort_values(["val_purchase_mrr", "val_ndcg@10"], ascending=False).iloc[0]
