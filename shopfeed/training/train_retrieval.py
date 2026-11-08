from __future__ import annotations

from typing import Dict

import pandas as pd

from shopfeed.config import Config
from shopfeed.features.batch import compute_batch_state
from shopfeed.retrieval.two_tower import RetrievalArtifacts, positive_pairs, train_two_tower


def fit_retrieval(tables: Dict[str, pd.DataFrame], cfg: Config, as_of: float, steps: int = None) -> RetrievalArtifacts:
    """User features and positive pairs both come from events strictly before `as_of`."""
    events = tables["events"][tables["events"].ts < as_of]
    state = compute_batch_state(events, tables["impressions"], tables["users"], tables["products"], tables["merchants"], as_of)
    pairs = positive_pairs(events)
    return train_two_tower(pairs, state.users, tables["products"], cfg.model, steps=steps)
