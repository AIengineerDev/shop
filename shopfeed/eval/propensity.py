"""Position examination propensity (spec §5.4, option B).

Under randomised traffic, position is independent of relevance, so
propensity(k) = CTR(k) / CTR(1) estimates P(examined | position k). The raw curve is made
monotone non-increasing with isotonic regression (weighted by impressions) and clipped at
config.PROPENSITY_CLIP, then versioned with the model in propensities.json.

The impressions.parquet `propensity` column is NOT an examination propensity: it is flat
across positions (~0.007 at every position) and is the logging policy's item-selection
probability. It is recorded for reference, not used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from shopfeed import config


def estimate_from_randomised(rows: pd.DataFrame, max_pos: int | None = None,
                             clip: float = config.PROPENSITY_CLIP) -> dict:
    g = rows.groupby("displayed_pos")["clicked"].agg(["mean", "size"]).sort_index()
    if max_pos:
        g = g.loc[:max_pos]
    raw = g["mean"] / g["mean"].iloc[0]
    iso = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0)
    smooth = iso.fit_transform(g.index.to_numpy(float), raw.to_numpy(), sample_weight=g["size"])
    smooth = smooth / smooth[0]
    return {
        "method": "randomised_ctr_ratio_isotonic",
        "positions": [int(p) for p in g.index],
        "raw": [round(float(x), 6) for x in raw],
        "propensity": [round(float(max(x, clip)), 6) for x in smooth],
        "clip": clip,
        "impressions_per_position": [int(n) for n in g["size"]],
    }


def lookup(curve: dict, positions) -> np.ndarray:
    """Propensity for 1-based positions; positions beyond the curve use its last value."""
    table = dict(zip(curve["positions"], curve["propensity"], strict=True))
    last = curve["propensity"][-1]
    return np.array([table.get(int(p), last) for p in positions], dtype=float)
