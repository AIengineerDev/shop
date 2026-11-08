"""Position-bias handling.

Training sees the real slot so the model can attribute part of the engagement to position.
At inference every candidate is scored as if it sat in slot 0, so position cannot leak into
the ranking decision (otherwise the ranker would just reproduce the logging policy's order).
"""
from __future__ import annotations

import numpy as np

from shopfeed.features.snapshot import POSITION_IDX

INFERENCE_POSITION = 0.0


def neutralize_position(X: np.ndarray) -> np.ndarray:
    Xn = np.array(X, copy=True)
    if Xn.ndim == 2 and Xn.shape[1] > POSITION_IDX:
        Xn[:, POSITION_IDX] = INFERENCE_POSITION
    return Xn
