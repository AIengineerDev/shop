"""Deterministic experiment bucketing and a bootstrap comparison of per-user metrics."""
from __future__ import annotations

import zlib
from typing import Dict

import numpy as np
import pandas as pd


def assign_bucket(user_id: str, experiment: str, treatment_fraction: float = 0.5) -> str:
    h = zlib.crc32(f"{experiment}:{user_id}".encode()) / 2**32
    return "treatment" if h < treatment_fraction else "control"


def compare(per_user: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> Dict[str, float]:
    """per_user: columns user_id, bucket ('treatment'|'control'), metric. Returns lift and 95% CI."""
    rng = np.random.default_rng(seed)
    t = per_user[per_user.bucket == "treatment"].metric.values.astype(float)
    c = per_user[per_user.bucket == "control"].metric.values.astype(float)
    if len(t) == 0 or len(c) == 0:
        raise ValueError("both buckets need users")
    lift = float(t.mean() - c.mean())
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = rng.choice(t, len(t)).mean() - rng.choice(c, len(c)).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"treatment_mean": float(t.mean()), "control_mean": float(c.mean()), "lift": lift,
            "ci_low": float(lo), "ci_high": float(hi), "n_treatment": int(len(t)), "n_control": int(len(c)),
            "significant": bool(lo > 0 or hi < 0)}
