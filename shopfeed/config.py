"""All tunable knobs live here. Nothing in the pipeline hardcodes a threshold."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

DAY = 86400.0
HOUR = 3600.0

# Ranker heads. Order matters: it is the column order of the label matrix.
HEADS: List[str] = ["click", "dwell", "save", "atc", "purchase", "hide", "return"]

# Event type -> ranker head it labels. Events not listed do not create labels.
EVENT_TO_HEAD: Dict[str, str] = {
    "view": "click",
    "dwell": "dwell",
    "save": "save",
    "add_to_cart": "atc",
    "purchase": "purchase",
    "hide": "hide",
    "not_interested": "hide",
    "return": "return",
}
POSITIVE_HEADS = {"click", "dwell", "save", "atc", "purchase"}
NEGATIVE_HEADS = {"hide", "return"}

CATEGORIES: List[str] = [
    "running_shoes", "socks", "insoles", "coffee", "mugs", "skincare", "cosmetics",
    "supplements", "pet_food", "pet_toys", "tshirts", "jeans", "headphones", "phone_cases",
    "candles", "home_decor",
]
# Items bought together across categories (not substitutes). Used by the complements source.
COMPLEMENTS: Dict[str, List[str]] = {
    "running_shoes": ["socks", "insoles"],
    "coffee": ["mugs"],
    "skincare": ["cosmetics"],
    "pet_food": ["pet_toys"],
    "jeans": ["tshirts"],
    "headphones": ["phone_cases"],
    "candles": ["home_decor"],
}
CONSUMABLE_CYCLE_DAYS: Dict[str, float] = {
    "coffee": 30.0,
    "skincare": 45.0,
    "supplements": 30.0,
    "pet_food": 21.0,
}
COUNTRIES: List[str] = ["US", "CA", "GB", "AU", "DE"]


@dataclass
class ValueWeights:
    """Final value = sum_k w_k * p_k. Purchase/ATC are scaled by expected order value / ref_price."""
    click: float = 0.05
    dwell: float = 0.10
    save: float = 0.30
    atc: float = 0.50
    purchase: float = 1.00
    hide: float = -2.00
    return_: float = -1.00
    ref_price: float = 50.0  # order value normaliser so a $500 item does not dominate

    def as_dict(self) -> Dict[str, float]:
        return {
            "click": self.click, "dwell": self.dwell, "save": self.save, "atc": self.atc,
            "purchase": self.purchase, "hide": self.hide, "return": self.return_,
        }


@dataclass
class SourceCaps:
    two_tower: int = 800
    followed: int = 300
    complements: int = 300
    replenish: int = 200
    fresh: int = 200
    trending: int = 200


@dataclass
class RerankConfig:
    page_size: int = 20
    max_consecutive_merchant: int = 2
    max_same_category_top10: int = 3
    freshness_floor: int = 2
    fresh_window_s: float = DAY
    exploration_slots: int = 2
    merchant_share_cap: float = 0.30
    light_cut: int = 300


@dataclass
class EligibilityConfig:
    repeat_block_s: float = 7 * DAY          # purchased or shown within this window -> ineligible
    max_item_age_s: float = 30 * DAY         # older than this -> ineligible unless evergreen


@dataclass
class ModelConfig:
    emb_dim: int = 32
    hidden: int = 64
    n_experts: int = 3
    content_dim: int = 8
    lr: float = 1e-2
    epochs: int = 8
    batch_size: int = 512
    seed: int = 0


@dataclass
class TrainingConfig:
    label_window_s: float = 7 * DAY
    val_fraction: float = 0.2


@dataclass
class GuardrailThresholds:
    max_hide_rate: float = 0.05
    max_return_rate: float = 0.15
    max_head_merchant_share: float = 0.30   # impression share of top 1% merchants
    min_fresh_share: float = 0.05


@dataclass
class Config:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    value: ValueWeights = field(default_factory=ValueWeights)
    caps: SourceCaps = field(default_factory=SourceCaps)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    eligibility: EligibilityConfig = field(default_factory=EligibilityConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    guardrails: GuardrailThresholds = field(default_factory=GuardrailThresholds)
    latency_budget_ms: float = 150.0

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"
