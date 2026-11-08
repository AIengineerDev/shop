"""Policy thresholds. A PolicyConfig is immutable and versioned; every decision records its version."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class PolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str = "v1"
    currency: str = "USD"
    window_days: int = Field(default=30, ge=0)
    auto_limit: Decimal = Field(default=Decimal("200.00"), ge=0)
    history_max_refunds: int = Field(default=3, ge=0)
    history_window_days: int = Field(default=90, ge=0)
    # abuse model: a score in [0, 1]; the rule can only tighten, never approve
    abuse_escalate_at: float = Field(default=0.8, ge=0, le=1)
    abuse_tighten_at: float = Field(default=0.5, ge=0, le=1)
    abuse_tighten_factor: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyConfig":
        return cls.model_validate(d)


DEFAULT_POLICY = PolicyConfig()
