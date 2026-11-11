"""Reason code -> human text. Templated; never an LLM."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from repricer.models import ReasonCode
from repricer.money import fmt

TEMPLATES: Mapping[ReasonCode, str] = {
    ReasonCode.MATCH_LOWEST: "Matched lowest eligible competitor price {lowest}.",
    ReasonCode.UNDERCUT: "Undercut lowest eligible competitor price {lowest} to {raw_target}.",
    ReasonCode.HOLD_STRATEGY: "Policy strategy is hold; price unchanged.",
    ReasonCode.TARGET_MARGIN: "Targeted configured margin over cost {cost}: {raw_target}.",
    ReasonCode.NO_COMPETITOR_DATA: "No eligible competitor offers; kept current price {current}.",
    ReasonCode.CLAMP_MAX_DECREASE: "Decrease capped by policy max_decrease to {bounded}.",
    ReasonCode.CLAMP_MAX_INCREASE: "Increase capped by policy max_increase to {bounded}.",
    ReasonCode.FLOOR_APPLIED: "Raised to margin floor {floor} (cost {cost} + min margin).",
    ReasonCode.LOW_STOCK_NO_DECREASE: "Stock {stock} at or below threshold; decrease suppressed.",
    ReasonCode.ROUNDED: "Rounded to policy price ending: {proposed}.",
    ReasonCode.SHIPPING_UNKNOWN: "Lowest competitor offer did not state shipping; treated as 0.",
    ReasonCode.NO_CHANGE: "Resulting price equals current price {current}; nothing to do.",
    ReasonCode.FLOOR_ABOVE_CURRENT: (
        "BLOCKED: margin floor {floor} is above current price {current}; SKU already under "
        "margin, needs human attention."
    ),
    ReasonCode.COST_ANOMALY: (
        "BLOCKED: cost {cost} is anomalous relative to current price {current}."
    ),
    ReasonCode.SINGLE_SOURCE_LOW_CONF: (
        "BLOCKED: only one eligible competitor offer and its confidence is below 70."
    ),
    ReasonCode.DELTA_EXCEEDS_HARD_CAP: (
        "BLOCKED: change of {delta_bp} bp exceeds the global hard cap."
    ),
    ReasonCode.PRICE_OUT_OF_RANGE: "BLOCKED: computed price {proposed} is outside the sane range.",
}


def explain(codes: list[ReasonCode], ctx: Mapping[str, Any]) -> str:
    money_keys = {"lowest", "raw_target", "bounded", "floor", "cost", "current", "proposed"}
    rendered: dict[str, Any] = {}
    for k, v in ctx.items():
        rendered[k] = fmt(v) if (k in money_keys and v is not None) else v
    parts = []
    for code in codes:
        template = TEMPLATES.get(code, code.value)
        try:
            parts.append(template.format(**rendered))
        except (KeyError, IndexError):  # pragma: no cover - template/ctx mismatch
            parts.append(code.value)
    return " ".join(parts)
