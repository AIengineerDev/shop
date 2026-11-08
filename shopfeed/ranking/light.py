"""Light ranker: logistic regression on ~12 cheap features. Exists only to cut 2k -> 300."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from shopfeed.features.snapshot import FEATURE_NAMES
from shopfeed.ranking.position import neutralize_position

LIGHT_FEATURES: List[str] = [
    "u_log_purchases", "u_shop_pay", "p_log_price", "p_log_age_days", "p_log_views_1h",
    "p_log_orders_1h", "p_log_purchases", "p_hide_rate", "m_return_rate", "m_review_score",
    "m_followed", "x_cat_affinity", "position",
]
LIGHT_IDX = np.array([FEATURE_NAMES.index(f) for f in LIGHT_FEATURES])


class LightRanker:
    def __init__(self) -> None:
        self.clf = LogisticRegression(max_iter=500, C=1.0)
        self.fitted = False

    def fit(self, X: np.ndarray, y_any_positive: np.ndarray) -> "LightRanker":
        if len(np.unique(y_any_positive)) < 2:
            raise ValueError("light ranker needs both classes")
        self.clf.fit(X[:, LIGHT_IDX], y_any_positive)
        self.fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            # Untrained fallback: popularity + affinity heuristic, so the pipeline runs before training.
            Xn = neutralize_position(X)
            return Xn[:, FEATURE_NAMES.index("p_log_purchases")] + Xn[:, FEATURE_NAMES.index("x_cat_affinity")]
        # Plain numpy inference from stored weights: no dependency on the sklearn object or its version.
        z = neutralize_position(X)[:, LIGHT_IDX] @ self.coef_ + self.intercept_
        return 1.0 / (1.0 + np.exp(-z))

    @property
    def coef_(self) -> np.ndarray:
        return self.clf.coef_.ravel()

    @property
    def intercept_(self) -> float:
        return float(self.clf.intercept_[0])

    def to_state(self) -> dict:
        if not self.fitted:
            raise ValueError("cannot serialise an unfitted light ranker")
        return {"coef": self.coef_, "intercept": np.array([self.intercept_]), "features": np.array(LIGHT_FEATURES)}

    @classmethod
    def from_state(cls, state: dict) -> "LightRanker":
        if list(state["features"]) != LIGHT_FEATURES:
            raise ValueError("light ranker feature list changed since the artifact was saved; retrain")
        lr = cls()
        lr.clf.coef_ = np.asarray(state["coef"], dtype=float).reshape(1, -1)
        lr.clf.intercept_ = np.asarray(state["intercept"], dtype=float)
        lr.clf.classes_ = np.array([0, 1])
        lr.fitted = True
        return lr

    def cut(self, candidates: pd.DataFrame, X: np.ndarray, n: int) -> pd.DataFrame:
        """Return the top-n candidates (fewer if fewer exist), ids preserved, with light_score."""
        s = self.score(X)
        out = candidates.copy()
        out["light_score"] = s
        out["row_idx"] = np.arange(len(out))
        out = out.sort_values("light_score", ascending=False, kind="stable").head(n)
        return out
