"""Guardrails: a model that wins the primary metric but trips one of these does not ship."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from shopfeed.config import DAY, GuardrailThresholds


@dataclass
class GuardrailResult:
    value: float
    threshold: float
    passed: bool
    direction: str  # "max" or "min"


def guardrail_report(page_log: pd.DataFrame, events: pd.DataFrame, thresholds: GuardrailThresholds,
                     fresh_window_s: float = DAY) -> Dict[str, GuardrailResult]:
    """page_log: impressions with columns impression_id, product_id, merchant_id, published_at, ts.
    events: events linked by impression_id."""
    n = max(len(page_log), 1)
    ev = events[events.impression_id.isin(page_log.impression_id)] if len(page_log) else events.iloc[0:0]
    hides = ev[ev.type.isin(["hide", "not_interested"])].impression_id.nunique()
    purchases = ev[ev.type == "purchase"].impression_id.nunique()
    returns = ev[ev.type == "return"].impression_id.nunique()
    hide_rate = hides / n
    return_rate = returns / purchases if purchases else 0.0

    if len(page_log):
        share = page_log.merchant_id.value_counts(normalize=True)
        top_k = max(1, int(np.ceil(0.01 * share.size)))
        head_share = float(share.head(top_k).sum())
        fresh_share = float(((page_log.ts - page_log.published_at.astype(float)) <= fresh_window_s).mean())
    else:
        head_share, fresh_share = 0.0, 0.0

    return {
        "hide_rate": GuardrailResult(hide_rate, thresholds.max_hide_rate, hide_rate <= thresholds.max_hide_rate, "max"),
        "return_rate": GuardrailResult(return_rate, thresholds.max_return_rate, return_rate <= thresholds.max_return_rate, "max"),
        "head_merchant_share": GuardrailResult(head_share, thresholds.max_head_merchant_share, head_share <= thresholds.max_head_merchant_share, "max"),
        "fresh_share": GuardrailResult(fresh_share, thresholds.min_fresh_share, fresh_share >= thresholds.min_fresh_share, "min"),
    }


def failed(report: Dict[str, GuardrailResult]) -> list:
    return [k for k, v in report.items() if not v.passed]
