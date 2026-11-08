"""Offline metrics. These gate a model; they never ship one."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from shopfeed.config import HEADS


def per_head_metrics(Y: np.ndarray, P: np.ndarray) -> Dict[str, Dict[str, float]]:
    out = {}
    for i, h in enumerate(HEADS):
        y, p = Y[:, i], np.clip(P[:, i], 1e-6, 1 - 1e-6)
        prev = float(y.mean()) if len(y) else float("nan")
        if len(np.unique(y)) < 2:
            out[h] = {"auc": float("nan"), "logloss": float("nan"), "prevalence": prev}
            continue
        out[h] = {"auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, p)), "prevalence": prev}
    return out


def dcg(gains: np.ndarray) -> float:
    if len(gains) == 0:
        return 0.0
    return float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))


def ndcg_at_k(pred: np.ndarray, gain: np.ndarray, groups: np.ndarray, k: int = 20) -> float:
    """Mean NDCG@k over groups (e.g. request_id). Groups with zero total gain are skipped."""
    df = pd.DataFrame({"pred": pred, "gain": gain, "g": groups})
    vals = []
    for _, grp in df.groupby("g"):
        if grp.gain.sum() <= 0:
            continue
        ranked = grp.sort_values("pred", ascending=False).gain.values[:k]
        ideal = np.sort(grp.gain.values)[::-1][:k]
        vals.append(dcg(ranked) / dcg(ideal))
    return float(np.mean(vals)) if vals else float("nan")


def calibration_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile table: count, mean predicted, observed rate. Counts sum to len(p)."""
    if len(p) == 0:
        return pd.DataFrame(columns=["bin", "count", "mean_pred", "obs_rate"])
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, n_bins)
    rows = [{"bin": i, "count": len(b), "mean_pred": float(p[b].mean()) if len(b) else np.nan,
             "obs_rate": float(y[b].mean()) if len(b) else np.nan} for i, b in enumerate(bins)]
    return pd.DataFrame(rows)


def ips_estimate(rewards: np.ndarray, logging_propensity: np.ndarray, target_propensity: np.ndarray,
                 clip: Optional[float] = 10.0, self_normalized: bool = True) -> float:
    """Counterfactual value of the target policy from logged data.

    w_i = pi_target(a_i|x_i) / pi_log(a_i|x_i). Items the target would never show get weight 0.
    Self-normalised IPS (SNIPS) trades a little bias for much lower variance; it is the default.
    """
    w = np.asarray(target_propensity, float) / np.clip(np.asarray(logging_propensity, float), 1e-9, None)
    if clip is not None:
        w = np.minimum(w, clip)
    r = np.asarray(rewards, float)
    if self_normalized:
        return float(np.sum(w * r) / max(np.sum(w), 1e-9))
    return float(np.mean(w * r))
