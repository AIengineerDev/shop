"""Project-wide constants. Spec references in comments are to docs/specs/product-ranking-model.md."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
FIXTURE_CSV = ROOT / "tests" / "fixtures" / "ranking_items.csv"
RANKING_CSV = DATA_DIR / "ranking_items.csv"

SEED = 42

# Labels (§3.1, §4)
ATTRIBUTION_WINDOW_S = 24 * 3600  # wishlist / ATC / purchase -> most recent impressing query
RETURN_WINDOW_S = 30 * 24 * 3600  # return within 30 d demotes grade 4 -> 2 (§4.3)
SESSION_GAP_S = 30 * 60  # new session after 30 min without a request

# NDCG gain tables (§4.2)
GAIN_PURCHASE_DOMINANT = [0, 1, 2, 5, 20]
GAIN_EXPONENTIAL = [0, 1, 3, 7, 15]

# AOV tier (§2 user_aov_tier)
AOV_LOOKBACK_S = 180 * 24 * 3600
AOV_TIERS = ("Low", "Mid", "High")
AOV_DEFAULT_TIER = "Mid"

# Data-readiness gates (§9). Production values; the 45-day sample overrides them
# via check_ranking_data.py --profile sample (see dev plan §6).
GATES = {
    "production": {"min_rows": 1_000_000, "min_groups": 50_000, "min_purchase_attribution": 0.95},
    "sample": {"min_rows": 100_000, "min_groups": 5_000, "min_purchase_attribution": 0.70},
}

# Popularity prior (§5.5)
UNPOPULAR_MIN_IMPRESSIONS = 200
UNPOPULAR_MAX_CTR = 0.01
NEW_ITEM_DAYS = 14
CATEGORY_TOP_N = 20

# Serving (§12)
MAX_CANDIDATES = 200
RANK_TIMEOUT_S = 0.030
RANKED_LIST_TTL_S = 30 * 60
QUERY_CACHE_TTL_S = 5 * 60
PROPENSITY_CLIP = 0.1
