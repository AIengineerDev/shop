"""Value function: weighted sum of head probabilities.

Purchase and add-to-cart are scaled by expected order value relative to a reference price, so
the feed prefers items that move GMV, with a clip so a single luxury item cannot dominate.
Hide and return carry negative weights: the model that learns to provoke is punished here.
Changing the weights changes the ordering with no retraining.
"""
from __future__ import annotations

import numpy as np

from shopfeed.config import HEADS, ValueWeights

_H = {h: i for i, h in enumerate(HEADS)}


def value_score(probs: np.ndarray, prices: np.ndarray, w: ValueWeights, price_clip=(0.2, 5.0)) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != len(HEADS):
        raise ValueError(f"probs must be (n, {len(HEADS)})")
    ov = np.clip(np.asarray(prices, dtype=float) / w.ref_price, *price_clip)
    wd = w.as_dict()
    v = np.zeros(len(probs))
    for h, i in _H.items():
        scale = ov if h in ("purchase", "atc") else 1.0
        v += wd[h] * scale * probs[:, i]
    # Expected return cost grows with order value too.
    v += wd["return"] * (ov - 1.0) * probs[:, _H["return"]] * probs[:, _H["purchase"]]
    return v
