# Spec: Product Ranking Model for Marketplace Search

| | |
|---|---|
| Status | Draft v2 |
| Owner | Alex Lavrenin |
| Date | 2026-11-11 (v1: 2026-11-09) |
| Dataset | `data/ranking_items.csv` (schema below) |
| Depends on | Existing `ranker_v1` artifacts (two-tower retrieval + multi-head heavy ranker, NDCG@20 = 0.440) |
| Dev plan | [`docs/plans/product-ranking-model.dev-plan.md`](../plans/product-ranking-model.dev-plan.md) |

### Changelog

| Version | Change |
|---|---|
| v2 (2026-11-11) | Dwell excluded from the spec entirely: grade 1 = any click, grade 2 = wishlist (§4.2). Funnel priority made explicit (§3.1). `user_aov_tier` defined as a personalisation feature (§2). Max-history item and seller features added (§5.1). Popularity prior (best sellers up, unpopular down) added (§5.5). New §12 Serving (indexing, pagination, caching, latency, SLA) and §13 Test plan. CSV header extended with `impression_ts`, `wishlisted`, `seller_id`, `category` (§2, §9). |
| v1 (2026-11-09) | Initial draft. |

## 1. Problem statement

Given a user, a request context, and a set of candidate products retrieved for a search query, produce an ordering of those candidates such that the item the user will **purchase** appears as high as possible. Purchase is the commitment the system is optimised for; clicks, wishlist adds and add-to-cart are intermediate signals used because purchases are sparse, never the objective in their own right.

This is a learning-to-rank (LTR) problem, not a classification problem. The unit of optimisation is the **query group** (all candidates shown for one `query_id`), and the objective is the quality of the ordering within each group.

### Acceptance criteria (from product)

- [ ] Data collection plan: what is logged, at what grain, with what retention.
- [ ] Label definition: pointwise targets and a graded relevance target, with justification.
- [ ] Feature set: query, item, user, context, and cross features, with leakage rules.
- [ ] Model candidates: baselines through to production ranker, with selection rationale.
- [ ] Training procedure: splits, loss, position-bias handling, hyperparameters, reproducibility.
- [ ] Offline evaluation: metrics, slices, and promotion gates.
- [ ] Online evaluation: experiment design, guardrails, and rollout.
- [ ] Funnel priority: Search → Impressed → Clicked → Wishlist → Add to cart → **Purchase**; purchase is the highest-priority commitment and dominates the gain table (§3.1, §4.2).
- [ ] Graded label: 0 no click, 1 click, 2 wishlist, 3 add to cart, 4 purchase (§4.2).
- [ ] Max-history item data (review rating, review count, price history) and seller data (seller → items → reviews, highest-rated) are features (§5.1).
- [ ] Ranking places best sellers (overall and per category) high and unpopular items low, without starving new items (§5.5).
- [ ] Serving covers indexing, pagination, caching, latency budget and SLA (§12).
- [ ] Test plan: unit, data, model, integration, performance and online tests, each with a pass criterion (§13).

### Success metrics

v1 is successful when every **ship gate** below is met at 100% rollout and holds through the 4-week holdback. Targets are relative to the current production ordering unless stated. Baselines marked TBD are read from the first week of the holdback and frozen in `metrics.json`.

#### Business metrics

| Metric | Definition | Baseline | v1 target | Gate |
|---|---|---|---|---|
| Purchase conversion per search session | sessions with ≥ 1 attributed purchase / search sessions | TBD | **+3% relative** | Ship gate (lower 95% CI > 0) |
| Revenue per search session | attributed GMV / search sessions | TBD | +3% relative | Ship gate (lower 95% CI ≥ 0) |
| Add-to-cart rate per search | ATC events / search sessions | ≈ 2.8% of impressions | +5% relative | Target |
| Search-to-first-purchase time | median positions scrolled or queries issued before purchase | TBD | −10% | Target |
| Query reformulation rate | sessions with a second query within 60 s and no click / search sessions | TBD | no increase > 2% relative | Guardrail |
| Zero-click rate | search pages with no click / search pages | TBD | no increase > 2% relative | Guardrail |
| 30-day return rate on search-attributed orders | returned units / purchased units | TBD | no increase | Guardrail (delayed read) |
| Merchant GMV concentration | share of search GMV to top 1% merchants | TBD | no increase > 1 pt | Guardrail |
| Mean price of top-10 results | mean `price` over positions 1–10 | TBD | within ±5% | Guardrail |
| Wishlist rate per search | wishlist (`save`) events / search sessions | TBD | +3% relative | Target |
| Best-seller coverage in top 10 | share of queries where ≥ 1 of the category's top-20 sellers appears in positions 1–10, when one is in the candidate set | TBD | ≥ production | Target |
| Long-tail exposure | share of top-10 impressions to items outside the top 20% by 30-day purchases | TBD | no drop > 10% relative | Guardrail (anti rich-get-richer, §5.5) |
| New-item exposure | share of top-10 impressions to items published < 14 days | TBD | no drop > 10% relative | Guardrail |

Twelve-month ambition (v1 → v3): +8–10% purchase conversion per search session, which at current traffic is the single largest lever on marketplace GMV available to the search team. Put the actual dollar figure here once the finance baseline is confirmed.

#### Technical metrics

| Metric | Definition | Baseline (`ranker_v1`) | v1 target | Gate |
|---|---|---|---|---|
| NDCG@10 (graded, purchase-dominant gain), test split | §8.1 | 0.440 at @20; @10 TBD | **≥ +2% relative** vs logged production order | Ship gate |
| NDCG@10 on randomised slice | §8.3 | TBD | ≥ +2% relative, CI excludes 0 | Ship gate |
| Purchase@10 recall | purchased items ranked in top 10 / purchased items | TBD | ≥ production | Ship gate |
| Purchase MRR | mean 1 / rank of the purchased item | TBD | ≥ +3% relative | Ship gate |
| Per-head AUC (click / ATC / purchase) | §8.1 | 0.64 / 0.71 / 0.73 | ≥ 0.70 / 0.75 / 0.78 | Target |
| Scoring latency, 200 candidates | p50 / p99 on serving CPU | TBD | ≤ 2 ms / ≤ 8 ms | Ship gate |
| End-to-end search latency p99 | request in → ranked list out | TBD | ≤ +10 ms vs production | Guardrail |
| Feature freshness | age of `hist_ctr_7d` and item aggregates at request time | TBD | ≤ 24 h | Guardrail |
| Train–serve skew | fraction of requests where logged features ≠ recomputed features | n/a | < 0.1% | Guardrail |
| Offline–online correlation | Spearman between offline NDCG@10 delta and online conversion delta across experiments | n/a | ≥ 0.6 after 5 experiments | Health |
| Training wall-clock, 90-day window | end-to-end pipeline | n/a | ≤ 2 h on one 16-core box | Operability |
| Reproducibility | re-run with same SHA + window reproduces `metrics.json` | n/a | exact to 3 decimals | Operability |
| Model availability | ranker responses served from the model (not fallback) | n/a | ≥ 99.9% | Operability |
| Fallback correctness | on timeout or error, serve `bm25` ordering | n/a | 100% of failures degrade, none 5xx | Operability |
| Ranking service SLA | successful ranked responses within 50 ms / all requests, monthly | n/a | ≥ 99.9% | Ship gate (§12.5) |
| Ranked-list cache hit rate (page ≥ 2) | page-2+ requests served from the ranked-list cache / page-2+ requests | n/a | ≥ 95% | Operability (§12.3) |
| Pagination consistency | page-2+ responses with an item already shown on an earlier page | n/a | 0% | Ship gate (§12.2) |
| Index freshness | age of price / stock / rating in the serving index | TBD | ≤ 15 min price & stock; ≤ 24 h aggregates | Guardrail (§12.1) |
| Wishlist logging coverage | wishlist events with an attributed `query_id` | 91% of all events carry `impression_id` today | ≥ 95% | Data gate (§9) |

#### Non-goals for the metrics

- CTR is **not** a success metric. It is reported as a secondary signal only. A ranker can raise CTR by promoting cheap, attractive items that do not convert; the graded label and the mean-price guardrail exist to prevent exactly that.
- Offline NDCG alone never ships a model. It qualifies a candidate for the A/B; the business gates decide.

### Out of scope for v1

- Candidate generation (retrieval). We rank what retrieval returns; the two-tower model in `ranker_v1` is the assumed source.
- Personalised re-ranking for diversity or merchant fairness (v2, see §10).
- Real-time feature streaming. v1 uses features materialised at request time from batch stores.

## 2. Dataset schema: `data/ranking_items.csv`

One row = one (query, candidate item) impression. Rows sharing a `qid` form one ranking group. Column names below are the **actual CSV header**; the brief's longer names are given in the second column and are used in prose throughout this spec.

| CSV column | Brief name | Type | Role | Notes |
|---|---|---|---|---|
| `session_id` | session_id | string | grouping | User visit. Used for session-level splits so one session never straddles train/val. |
| `qid` | query_id | int | **group key** | The LTR `qid`. One search result page. |
| `query_text` | query_text | string | context | Raw query. Not fed directly to GBDT; used to derive query features and for neural rankers. |
| `user_id` | user_id | string | grouping / feature key | Join key to user features. Used for user-disjoint evaluation. |
| `item_id` | item_id | string | feature key | Join key to item features. |
| `device` | device_type | categorical | context | Observed values: `mobile`, `desktop`. |
| `user_aov_tier` | user_aov_tier | categorical {Low, Mid, High} | user / **personalisation** | **Average Order Value** tier: the user's typical basket value, bucketed by tercile of trailing-180-day AOV over all users. It is *not* a "best value user" or loyalty score. It is the main personalisation input in v1: it tells the ranker what price band this user actually buys in, so a `High` user sees premium items earlier and a `Low` user is not shown items they will not buy. Used via the `price × aov_tier` cross feature (§5.1). Users with no orders get `Mid` plus `is_new_user = 1`. |
| `bm25` | bm25_score | float | relevance | **Best Matching 25**: lexical query–item match from the search index (term frequency, inverse document frequency, length-normalised). Observed range 7.5–16.2. Not comparable across queries; use within-group rank (§5.1). |
| `semantic_sim` | semantic_sim | float | relevance | Query-item cosine **similarity**, higher = closer. Confirmed from the sample: 0.88–0.91 on purchased items, 0.52–0.61 on bottom negatives. The brief's wording "cosine distance" is wrong; no sign flip needed. |
| `price` | price | float | item | Item price at impression time. |
| `hist_ctr_7d` | historical_ctr_7d | float | item | Trailing 7-day CTR for the item. Must be computed from data strictly *before* the impression timestamp. |
| `rating` | review_rating | float | item | Mean review rating. |
| `displayed_pos` | displayed_position | int | **logging bias** | Rank position at which the item was shown, 1-based. Used only for debiasing; **never** a model input at serving time. |
| `clicked` | clicked | 0/1 | label | |
| `added_to_cart` | added_to_cart | 0/1 | label | |
| `purchased` | purchased | 0/1 | label | |
| `relevance_grade` | relevance_grade | int 0–4 | **label** | 0 no click; 1 click; 2 wishlist; 3 add-to-cart; 4 purchase. Derived, see §4.2. |
| `impression_ts` | — | float (unix s) | ordering | From `impressions.parquet.ts`. Required for temporal splits and point-in-time features. |
| `wishlisted` | — | 0/1 | label | From `events.type = save`. |
| `seller_id` | sellers ID | string | feature key | `merchant_id` in the parquet sources. Join key for seller features (§5.1). |
| `category` | — | categorical | feature key / slice | From `products.parquet.category`. Needed for per-category best-seller rank (§5.5) and slices. |

### Current state

As of 2026-11-09 `data/ranking_items.csv` is a **10-row sample fixture**: 2 sessions, 2 query groups (`wireless headphones`, `mechanical keyboard`), 5 candidates each, one purchase per group. It is sufficient to pin the schema and to unit-test the feature builder, label checks and NDCG implementation. It is **not** training data. The full table must be regenerated at scale from the sibling parquet files (`impressions.parquet`, `events.parquet`, `products.parquet`, `users.parquet`) using this header as the contract.

Observations from the sample that the full dataset must preserve:
- Label consistency holds on all 10 rows (`purchased ⇒ grade 4`, `added_to_cart ⇒ grade ≥ 3`, `clicked = 0 ⇒ grade 0`).
- `displayed_pos` is a clean permutation of 1..5 within each group.
- Labels are strongly position-correlated (all purchases at position 1), which is exactly the logging bias §5.4 addresses. The sample is too small to estimate propensities from.
- Groups are tiny (5 candidates). Production groups should carry the full result page (20–100 candidates); a 5-item cap would throw away most negatives.

### Columns the CSV is missing that we need

`impression_ts`, `wishlisted`, `seller_id` and `category` were on this list in v1. They are now in the header above because the parquet sources already have them (`impressions.ts`, `events.type = save`, `products.merchant_id`, `products.category`). Still missing:

| Column | Why |
|---|---|
| `candidate_set_size` | Normalising position; detecting truncated result pages. |
| `query_intent` / `is_navigational` | Query-type slicing; navigational queries have degenerate label distributions. |
| `shipping_eta_days` | Strong purchase predictor. `merchants.fulfilment_days` is a seller-level proxy until then. |

`in_stock` exists in `products.parquet` but is a current snapshot, not point-in-time, so it is a serving filter (§12.1), not a training feature, until it is snapshotted per impression.

Request the rest from the logging pipeline. Train v1 without them if they cannot be backfilled; add in v1.1.

### Source data facts (as of 2026-11-11)

| Source | Rows | Notes |
|---|---|---|
| `impressions.parquet` | 196,420 | 9,821 requests; 2025-05-01 → 2025-06-15 (45 days, not 90); has `ts`, `position` (0-based), `propensity`; **100% `source = logging_random`** |
| `events.parquet` | 58,960 | types: view 23,069 · save 5,654 · add_to_cart 5,255 · purchase 4,510 · follow 4,048 · hide 2,524 · return 635 |
| `products.parquet` | 4,000 | `merchant_id`, `category`, `price`, `in_stock`, `published_at` |
| `merchants.parquet` | 60 | `review_score`, `return_rate`, `fulfilment_days`, `popularity` |
| `users.parquet` | 2,000 | no AOV tier column: derive from purchase `value` (§2 `user_aov_tier`) |

Consequences: (1) the whole log is randomised traffic, so propensities can be estimated directly (§5.4 option B, no EM needed) but there is no production-ordered baseline to compare against: "logged order" in §6.1 means the randomised order. (2) No `query_text` in impressions: the source is a feed, not search. Query features come from the search service log (§3.1) once it exists; the 10-row CSV is the only query-bearing data today. (3) `hide` demotes click-only rows (§4.3); `return` is not yet applied (§4.3). (4) `users.parquet` gains `aov_tier`, `aov_value`, `aov_order_count`, `aov_asof_ts` via `scripts/build_user_aov.py` (serving snapshot only; training uses the point-in-time tier).

### Build mapping (feed log → CSV), implemented in `shopfeed/data/build.py`

| Column | Source / rule |
|---|---|
| `qid` | integer part of `request_id` |
| `session_id` | per user, new session when a request starts > 30 min after the previous one |
| `displayed_pos` | `position + 1`. Duplicate (request, item) impressions (2,568 rows) are merged: earliest position kept, labels = max. This leaves documented position gaps in 1,096 groups. |
| `clicked` / `wishlisted` / `added_to_cart` / `purchased` | `view` / `save` / `add_to_cart` / `purchase` events (see attribution, §3.1) |
| `user_aov_tier` | point-in-time tier as of the impression day's start (`shopfeed/data/aov.py`) |
| `semantic_sim` | cosine(user `t0..t7`, product `c0..c7`): **user–item proxy**, not query–item |
| `bm25`, `rating` | NaN (not in the feed log; exempt from the §9 null-rate check) |
| `query_text` | empty (feed mode; §9.1 query rules apply only to non-empty queries) |
| `device` | `unknown` (added to the §9.1 allowed set) |

## 3. Data collection

### 3.1 Event logging

Log at the impression grain with a shared `query_id` so that every candidate rendered for a search result page is captured, including non-clicked ones. Negatives are the majority of the signal; a click-only log is useless for LTR.

**Funnel priority.** Events are ordered by commitment. Logging completeness, alerting and label weight all follow this order, and purchase is the highest priority: a missing purchase event corrupts a grade-4 label, which is worth 20 clicks in the gain table (§4.2).

| Priority | Event | Fields | Source | Grade it feeds | Max loss before alert |
|---|---|---|---|---|---|
| 1 (highest) | `purchase` | order_id, item_id, user_id, **query_id (attributed)**, price_paid, ts | backend | 4 | 0.1% |
| 2 | `add_to_cart` | query_id (attributed), item_id, ts | backend | 3 | 0.5% |
| 3 | `wishlist_add` | query_id (attributed), item_id, ts | backend | 2 | 1% |
| 4 | `item_clicked` | query_id, item_id, ts | client | ≥ 1 | 2% |
| 5 | `item_impressed` | query_id, item_id, displayed_position, ts, viewport_visible (bool) | client | 0 (negative) | 2% |
| 6 | `search_issued` | query_id, session_id, user_id, query_text, device_type, ts, candidate_set_size, page, page_size | search service | group key | 0.1% |

`wishlist_add` maps to `events.type = save` in the current parquet.

Attribution: backend events carry the attributed impression / `query_id` at write time, and a stamp is **trusted** up to a 7-day sanity cap (`STAMPED_MAX_LAG_S`). Only events without a stamp are reconstructed: attributed to the user's most recent impression of the item within 24 h (`ATTRIBUTION_WINDOW_S`). Events that match no impression are non-search and excluded.

Why not a hard 24 h window on stamped events: in the current log, stamped purchases arrive up to 48 h after the impression (median 24.8 h). 1,719 of 3,353 (51%) are beyond 24 h, so a 24 h cut would discard half of grade 4. The window changes label prevalence materially; `reports/build_report.json` records the lag counts on every build.

### 3.2 Feature snapshots

Item-level aggregates (`historical_ctr_7d`, `review_rating`, price) must be **snapshotted at impression time** and logged alongside the impression, or recomputed point-in-time from an event store. Joining today's item table onto last month's impressions leaks future information into training.

### 3.3 Volume, retention, sampling

- Target: ≥ 90 days of impressions, split 60 / 20 / 20 by time into train / validation / test (§7.1).
- Keep all query groups that contain at least one positive (grade ≥ 1). For groups with zero positives, keep a 10% random sample; they contribute nothing to pairwise/listwise loss but are needed for calibration and for pointwise baselines.
- Within a group, keep all candidates. Do not subsample negatives inside a group; it distorts position-bias estimation.
- Raw events: 13 months retention (year-over-year seasonality). Training tables: 6 months.

### 3.4 Randomised exploration traffic

Allocate 1–2% of traffic to a randomised-top-k policy (shuffle positions 1–10 for the production ranker's top 10). This is the only clean way to estimate position bias (§5.4) and gives unbiased data for off-policy evaluation (§7.3). Without it, every model we train inherits the biases of the ranker that produced the logs.

## 4. Labels

### 4.1 Pointwise binary targets

`clicked`, `added_to_cart`, `purchased`. Used for:
- Multi-task auxiliary heads (improves representation learning on the sparse purchase signal).
- Calibrated probability outputs that downstream systems (ads, promotions) can consume.
- Sanity-check baselines.

Expected prevalence (from `ranker_v1` metrics): click ≈ 12%, ATC ≈ 2.8%, purchase ≈ 1.8%. Purchase alone is too sparse to train a ranker on directly; it is the reason the graded label exists.

### 4.2 Graded relevance target (primary)

`relevance_grade` ∈ {0,1,2,3,4} is the primary training label for the LTR objective.

| Grade | Definition | Rationale |
|---|---|---|
| 0 | Impressed, not clicked | Negative. Note: a non-viewport-visible impression is a weaker negative; drop or down-weight if `viewport_visible` is available. |
| 1 | Click | Interest signal only. Deliberately low gain so the model is not rewarded for attractive-but-wrong items. |
| 2 | Wishlist add | Stated consideration without commitment. |
| 3 | Add to cart | Strong purchase intent. |
| 4 | Purchase | Terminal success. |

Gain function for NDCG: **purchase-dominant**, `gain = {0, 1, 2, 5, 20}` for grades 0–4. Purchase is the commitment the ranker is optimised for: one purchase outweighs four add-to-carts or twenty clicks, so a model cannot "win" a query by stacking clicks above the item the user actually bought. The standard `2^g − 1` table ({0,1,3,7,15}) is kept as a reported secondary metric for comparability with the literature, but it is not the training target. Revisit the gain table if the business wants revenue weighting (`gain = 2^g − 1` × `price` normalised), but do not do that in v1 because it conflates relevance with price and will push expensive items up regardless of fit.

### 4.3 Label rules

- Grades are monotone in the funnel: a purchased item is grade 4 regardless of other signals.
- Returns/refunds within 30 days: demote grade 4 → 2 at label-build time when the return signal is available (`ranker_v1` already logs a `return` head). Not in v1 CSV; add when `order` data is joined.
- Items clicked but never impressed (deep-link, external) are excluded from the group.
- `relevance_grade` must be consistent with the binary columns: `purchased=1 ⇒ grade=4`, `added_to_cart=1 ⇒ grade≥3`, `wishlisted=1 ⇒ grade≥2`, `clicked=0 ⇒ grade=0`. Assert this in the data-readiness check.
- A wishlist, ATC or purchase without a click (e.g. add-to-cart from the result card) still sets the grade; set `clicked = 1` for consistency and count these rows in the build report.
- `hide` events: a click-only row that is hidden becomes grade 0. Wishlist, ATC and purchase outrank a hide (so the implications above always hold). Do not invent a negative grade; LambdaMART gains must be ≥ 0.

### 4.4 Why not implicit-only (click) labels

Click-only labels over-reward items with attractive thumbnails and low prices and under-reward items that convert. The graded label with a purchase-dominant gain aligns the ranking objective with the funnel endpoint while keeping enough density (any click gives grade ≥ 1) to train.

## 5. Features

### 5.1 Feature groups

| Group | Features (v1, from CSV) | Additions for v1.1 |
|---|---|---|
| **Query–item relevance** | `bm25_score`, `semantic_sim` | title/attribute exact-match flags, query-category match, query term coverage |
| **Item quality / popularity** | `historical_ctr_7d`, `review_rating`, `price` | review_count, 7d ATC rate, shipping_eta_days, return rate |
| **Item max history** (v2) | `item_review_count`, `item_rating_smoothed` (Bayesian: `(n·r + m·C) / (n + m)`, `m = 20`, `C` = category mean), `item_purchases_30d`, `item_purchases_all` (all history up to `t`), `item_wishlist_30d`, `price_vs_item_median_90d` (current / own median), `price_is_item_min_90d`, `item_age_days` | price-drop recency |
| **Seller** (v2) | `seller_review_score`, `seller_return_rate`, `seller_fulfilment_days`, `seller_item_count`, `seller_purchases_30d`, `seller_top_item_rating` (highest `item_rating_smoothed` among the seller's items with ≥ 20 reviews), `seller_review_weighted_rating` (mean item rating weighted by review count), `user_follows_seller` | seller tenure, seller dispute rate |
| **Popularity prior** (v2, §5.5) | `item_bestseller_rank_category` (rank by 30-day purchases within `category`, 1 = best), `item_bestseller_pct_global`, `is_category_top20`, `is_unpopular` (0 purchases and < 1% CTR over 30 days with ≥ 200 impressions) | — |
| **User** | `user_aov_tier`, `is_new_user` | user 30d purchase count, user category affinity vector (top-k categories), user price sensitivity (median purchased price) |
| **Context** | `device_type` | hour-of-day, day-of-week, country/region, query length, query intent class |
| **Cross** | `price / user_aov_tier median price` (price fit), `price` rank within group, `bm25` rank within group, `semantic_sim` z-score within group | user-category × item-category affinity, user's past engagement with this merchant |
| **Group-level** | candidate_set_size, mean/max bm25 in group, mean/max semantic_sim in group | query popularity (searches / 7d), query historical CTR |

Within-group normalised features (rank, z-score, ratio-to-max) matter a lot for GBDT rankers: raw `bm25_score` is not comparable across queries, but "this item has the top BM25 in its group" is.

**"Max historical" means all history up to the impression, never after.** Every item and seller aggregate is computed over `[first_event, t)` for an impression at `t` (or the stated window ending at `t`). The current `products.parquet` / `merchants.parquet` scores (`review_score`, `popularity`) are present-day snapshots; using them directly on last month's impressions is leakage. Until snapshots exist, recompute from `events.parquet`, and treat the merchant snapshot columns as v1-acceptable only for slow-moving fields (`fulfilment_days`, `return_rate`) with a leakage note in `features.json`.

**Implementation status (M2, `shopfeed/features/`).** 42 features. Implemented: all v1 CSV features, item history (`item_purchases_30d`, `item_purchases_all`, `item_wishlist_30d`, `item_age_days`), seller (`seller_return_rate`, `seller_fulfilment_days`, `seller_item_count`, `seller_purchases_30d`, `user_follows_seller` from timestamped `follow` events), user (`user_order_count`, `is_new_user`, `user_aov_value`), popularity prior (§5.5, as of day start), cross `price_to_user_aov` (price ÷ user's point-in-time AOV), and within-group rank / z-score / ratio-to-max for `bm25`, `semantic_sim`, `price`, `hist_ctr_7d`, `item_purchases_30d`. **Not available in the sources, omitted:** item reviews (`item_review_count`, `item_rating_smoothed`, seller rating aggregates) and price history (purchase `value` always equals list price). `merchants.review_score` / `popularity` are excluded as present-day snapshots. On the current sample `is_unpopular` is 0 everywhere (no item reaches 200 impressions in 30 days without a sale).

### 5.2 Features explicitly excluded from the model

| Feature | Reason |
|---|---|
| `displayed_position` | Logging bias. Only used as a debiasing input during training (§5.4). At serving time position does not exist yet. |
| `clicked`, `added_to_cart`, `purchased`, `wishlisted`, `relevance_grade` | Labels. |
| `seller_id`, `category` as raw IDs | Memorisation. Use them only as join keys and through the aggregates in §5.1. `category` may be a LightGBM categorical feature if it has ≤ 200 levels. |
| `session_id`, `query_id`, `user_id`, `item_id` as raw IDs | Memorisation. Item ID embeddings are allowed only in the neural ranker with dropout and a frequency floor. |
| Any aggregate computed with a window that includes the impression time | Leakage. |

### 5.3 Leakage checks

- `historical_ctr_7d` for an impression at time `t` must use clicks in `[t − 7d, t)`. Verify by recomputing from `events.parquet` for a sample and comparing.
- Feature importances: if a feature has implausibly high importance (> 50% gain) on the first fit, assume leakage until proven otherwise.
- Train on the earliest 60% of sessions, evaluate on the latest 20% (§7.1). If test NDCG is far above the online metric, suspect leakage.

### 5.4 Position bias handling

Logged clicks are biased toward low positions. Three options, in increasing complexity. v1 uses option B.

| Option | Method | Pros / cons |
|---|---|---|
| A. Position as a training-only feature | Add `displayed_position` as a feature; at inference set it to a constant (e.g. 1). | Simple. Works for GBDT. Assumes additive bias. |
| B. Inverse propensity weighting (IPW) | Estimate `P(examined | position)` from randomised traffic (§3.4) or via EM on the logs (regression-EM / intervention harvesting). Weight each clicked example by `1 / propensity`. Use LambdaMART's unbiased variant (`lambdarank_unbiased` in LightGBM). | Principled. Requires propensity estimates; high-variance at deep positions, so clip propensities at 0.1. |
| C. Two-tower examination model | Jointly learn examination (position-conditioned) and relevance (feature-conditioned) towers, use only the relevance tower at serving. | Most flexible. Needs the neural ranker. v2. |

Propensity curve estimation: with 1–2% randomised traffic, `propensity(k) = CTR_random(k) / CTR_random(1)`. Store as a versioned artefact next to the model. The current log is 100% `logging_random` and already carries a per-row `propensity`: estimate the curve from it directly and cross-check against the logged value.

### 5.5 Popularity prior: best sellers up, unpopular items down

Product requirement: within a query's candidates, **best sellers (overall and per category) rank near the top and unpopular items near the bottom**.

**How**: as learned features, not hard rules. The §5.1 popularity-prior features plus the purchase-dominant gain already push best sellers up, because best sellers are by definition the items that earn grade 4. The model learns how much popularity to trust per query; a hard "pin best sellers to the top" rule would put a best-selling phone case above the phone the user searched for.

**Business rules on top of the model** (post-ranking, in `shopfeed/serving/ranker.py`, each behind a flag):

| Rule | Behaviour | Default |
|---|---|---|
| Relevance floor | An item is eligible for positions 1–3 only if its `semantic_sim` is ≥ the group's median. Stops popularity from overriding relevance. | on |
| Best-seller boost | If the category's #1 seller is in the candidate set and the model put it below position 10, move it to position 10. | off; enable only if the A/B shows best-seller coverage below production |
| Unpopular demotion | Items with `is_unpopular = 1` cannot occupy positions 1–5. They are not removed. | on |
| New-item exploration | Items with `item_age_days < 14` and < 200 impressions are exempt from unpopular demotion and get one slot in positions 6–10 per page, round-robin. | on |

**Why the exploration slot is not optional**: "unpopular" is mostly "unseen". Without it, popularity features create a feedback loop where new and long-tail items never get impressions, never get purchases, and stay at the bottom permanently. The long-tail and new-item exposure guardrails in §1 measure this.

**Offline check**: report NDCG@10 and Purchase MRR separately for groups where the purchased item is a category top-20 seller vs not. A model that wins only on best sellers is just a popularity sort; compare against baseline #2 (§6.1).

## 6. Models

### 6.1 Candidates

| # | Model | Role | Expected NDCG@10 vs baseline |
|---|---|---|---|
| 0 | Current production order (as logged) | Reference point | 0 |
| 1 | Sort by `bm25_score` | Lexical baseline | negative |
| 2 | Sort by `historical_ctr_7d` | Popularity baseline (clicks) | ~ 0 |
| 2a | Sort by `item_purchases_30d` (best-seller sort) | Popularity baseline (purchases); the bar §5.5 must clear | + small |
| 3 | Pointwise GBDT (LightGBM, binary on `clicked`) | Sanity baseline | + small |
| 4 | **LambdaMART (LightGBM `lambdarank`, graded label, IPW)** | **v1 production candidate** | + 5–10% |
| 5 | Multi-task neural ranker (MLP over same features + item ID embedding, heads for click/ATC/purchase + ranking head) | v1.1 / v2 if embeddings and real-time features pay off | + on top of #4 |
| 6 | Cross-encoder (query_text × item text) as a feature into #4 | Relevance feature upgrade | additive to #4 |

### 6.2 Recommendation: LambdaMART for v1

- Tabular, ~15–40 dense features, graded labels, group structure: this is the regime where GBDT LTR is strongest and hardest to beat.
- Trains in minutes on tens of millions of rows; no GPU.
- Native support for `qid` groups, graded labels, and unbiased lambdarank in LightGBM.
- Interpretable feature importance, which matters for the leakage checks in §5.3.
- Latency: scoring 200 candidates with 500 trees is < 2 ms on CPU.

The neural ranker (#5) is justified only when we add high-cardinality sparse features (item ID, user-history sequences, text) that trees cannot exploit. `ranker_v1` already has a multi-head heavy model; its per-head AUCs (0.64–0.73) are modest, which suggests the feature set, not the architecture, is the bottleneck. Fix features first.

### 6.3 Loss

- LambdaMART: LambdaRank gradients, optimising NDCG@10 directly. `eval_at = [5, 10, 20]`.
- Neural ranker: ListNet / softmax cross-entropy over the group with graded gains, plus BCE auxiliary heads with loss weights {click: 0.25, atc: 0.5, purchase: 1.0}, so the purchase head dominates the auxiliary signal.

## 7. Training

### 7.1 Splits

**60 / 20 / 20 by time**, partitioned on `session_id`, with a user-disjoint check. Sessions are ordered by their first `impression_ts`; the earliest 60% of sessions form train, the next 20% validation, the latest 20% test. Proportions are measured in **sessions**, not rows, so a few very large sessions cannot skew the split.

| Split | Share of sessions | Time order | Purpose |
|---|---|---|---|
| Train | 60% | earliest | fit |
| Validation | 20% | middle | early stopping, hyperparameter and gain selection (§7.3, M7) |
| Test | 20% | latest | single final read; never tuned on |

On a 90-day window this is roughly days 1–54 / 55–72 / 73–90. Cut points are stored in `meta.json` as timestamps so the split is reproducible without recomputing percentiles.

- Groups must not be split across partitions. Partition on `session_id`, which implies `query_id` integrity.
- Cut strictly on time, never randomly within the 60/20/20 shares. The ordering is what protects against leakage; the shares only set how much data each partition gets.
- Validation and test are the same size by design so that the number chosen on validation and the number reported on test have comparable variance.
- Report test metrics separately on users seen in train vs users unseen (cold-start slice).
- Never shuffle-split. Users and items are heavily repeated; a random split inflates metrics by 10–20% from memorisation.

### 7.2 Group filtering

- Drop groups with < 2 candidates (no ranking signal).
- Drop groups where all grades are equal (zero pairwise gradient) from the *ranking* training set; keep them for pointwise heads.
- Cap groups at 100 candidates by keeping all positives and a random sample of negatives, only if memory becomes a constraint.

### 7.3 LightGBM configuration (starting point)

```text
objective          = lambdarank
metric             = ndcg
eval_at            = 10,5,20   # NDCG@10 first: early stopping uses the first metric only
label_gain         = 0,1,2,5,20   # purchase-dominant, see §4.2
lambdarank_truncation_level = 20
num_leaves         = 63
learning_rate      = 0.05
min_data_in_leaf   = 100
feature_fraction   = 0.8
bagging_fraction   = 0.8, bagging_freq = 1
lambda_l2          = 1.0
num_boost_round    = 2000, early_stopping_rounds = 100
# position debiasing
# LightGBM >= 4.1 has no `lambdarank_unbiased` flag: unbiased LambdaMART is enabled by
# passing Dataset(position=displayed_position - 1) (training only).
lambdarank_position_bias_regularization = 1.0
```

Tune `num_leaves`, `learning_rate`, `min_data_in_leaf`, `feature_fraction` with ~30 trials of random search on the validation split. Do not tune on test.

### 7.4 Reproducibility and artefacts

Each training run writes to `artifacts/ranker_vN/`:

| File | Content |
|---|---|
| `model.txt` | LightGBM model dump |
| `features.json` | Ordered feature list, dtypes, categorical vocabularies, within-group normalisation spec |
| `propensities.json` | Position propensity curve used for IPW |
| `metrics.json` | Validation and test metrics, per slice |
| `meta.json` | Data window, git SHA, seed, row/group counts, label gain table |
| `importance.csv` | Gain and split importance per feature |

Seed everything. The data window and git SHA together must reproduce `metrics.json` to 3 decimals.

### 7.5 Retraining cadence

Weekly full retrain on a rolling 90-day window. Trigger an off-cycle retrain if the 7-day online NDCG proxy drifts > 3% relative or if the feature distribution monitor (PSI > 0.2 on any top-10 feature) fires.

## 8. Evaluation

### 8.1 Offline metrics

| Metric | Why | Gate |
|---|---|---|
| **NDCG@10** (graded, purchase-dominant gain {0,1,2,5,20}) | Primary. Matches training objective and page-one visibility. | Must beat production ordering (logged `displayed_position`) by ≥ 2% relative on test. |
| NDCG@5, NDCG@20 | Top-of-page and full-page sensitivity | Report |
| NDCG@10 with standard gain `2^g − 1` | Comparability with literature and `ranker_v1` | Report |
| **Purchase MRR** | 1 / rank of the purchased item, averaged over groups with a purchase | Must beat production; co-primary with NDCG@10 |
| MAP (grade ≥ 2 as relevant) | Robustness check on a binarised view | Report |
| Purchase@1 / Purchase@3 / Purchase@10 | Fraction of purchased items landing at position 1, top 3, top 10 | Purchase@10 ≥ production; Purchase@1 and @3 report |
| Per-head AUC / logloss (click, ATC, purchase) | Calibration of auxiliary outputs if used downstream | Report |
| Mean price of top-10 | Guard against the model learning "cheap wins" | Within ±5% of production |

### 8.2 Slices

Report every metric on: `device_type`, `user_aov_tier`, query length bucket (1 / 2–3 / 4+ terms), new vs returning user, candidate set size bucket, top-20 categories, seller size bucket (by `seller_item_count`), and purchased-item popularity (category top-20 seller vs long tail, §5.5). A model that wins overall but loses on `High` AOV tier or mobile is not shippable without discussion.

### 8.3 Unbiased offline evaluation

Logged NDCG rewards models that agree with the logging policy. Two corrections:

1. **IPW-weighted NDCG** using the propensity curve from §5.4. Report alongside raw NDCG.
2. **Off-policy evaluation on the randomised slice** (§3.4): on randomised traffic, position-conditioned bias is removed by construction, so NDCG on that slice is the most trustworthy offline number. It is small (1–2% of traffic), so report confidence intervals via bootstrap over query groups.

### 8.4 Statistical rigour

- Bootstrap 1,000 resamples over `query_id` for every metric; report mean and 95% CI.
- Paired comparison against production: compute per-query deltas and a paired t-test / Wilcoxon. Promote only if the lower CI bound of the delta is > 0.

### 8.5 Online evaluation

Targets and gates are defined once in §1 *Success metrics*; this section describes how they are measured.

**Design**: user-level A/B, 50/50 or 90/10 ramp, minimum 2 weeks (covers weekly seasonality), traffic sized for 80% power to detect a 1% relative lift in purchase conversion per search.

| Metric | Type | Decision rule |
|---|---|---|
| Purchase conversion per search session | **Primary** | Must be ≥ control, p < 0.05 |
| Revenue per search session | Primary (business) | Must be ≥ control; report CI |
| CTR@10, ATC rate | Secondary | Expect lift; informational |
| Query reformulation rate, zero-click rate | Guardrail | Must not increase > 2% relative |
| Search latency p99 | Guardrail | ≤ +10 ms |
| Return rate (30-day, delayed read) | Guardrail | Must not increase |
| Merchant concentration (share of GMV to top 1% merchants) | Guardrail | Must not increase > 1 pt |

**Interleaving**: run team-draft interleaving against production for 3 days before the A/B as a cheap, high-sensitivity pre-check. Interleaving needs ~10× less traffic than A/B for ranking comparisons and catches obviously broken models early.

**Rollout**: 1% → 10% → 50% → 100%, each stage ≥ 3 days with guardrails green. Keep a 5% holdback for 4 weeks after 100% to measure long-term effects (novelty decay, return rate).

### 8.6 Offline–online consistency

Track the correlation between offline NDCG@10 delta and online purchase-conversion delta across experiments. If after 5+ experiments the correlation is weak, revisit the gain table (purchase weight first) and the position debiasing, since the offline metric is not measuring what the business cares about.

## 9. Data-readiness checks (run before any training)

Implemented as a script that fails loudly. All must pass.

- [ ] Header matches §2 exactly (`session_id,qid,query_text,user_id,item_id,device,user_aov_tier,bm25,semantic_sim,price,hist_ctr_7d,rating,displayed_pos,clicked,added_to_cart,purchased,relevance_grade,impression_ts,wishlisted,seller_id,category`). The 10-row fixture is regenerated with the four v2 columns at M1.
- [ ] Wishlist and purchase attribution: ≥ 95% of `save` and `purchase` events resolve to a `query_id`.
- [ ] Every `seller_id` and `category` joins to `products.parquet` / `merchants.parquet` (0 orphans).
- [ ] Row count ≥ 1M; group count ≥ 50K (the current 10-row fixture fails this by design).
- [ ] No nulls in group keys or labels. Feature null rate < 5% per column; nulls encoded as NaN, not 0.
- [ ] Label consistency: `purchased ⇒ grade 4`, `added_to_cart ⇒ grade ≥ 3`, `clicked = 0 ⇒ grade 0`.
- [ ] Every group has ≥ 2 rows; `displayed_position` within a group is a permutation of `1..n` (or documented gaps).
- [ ] `semantic_sim` direction: positive correlation with `clicked` (confirmed on the fixture; re-check at scale).
- [ ] `historical_ctr_7d` point-in-time check against `events.parquet` on a 1K sample.
- [ ] Position bias visible: CTR at position 1 > CTR at position 10 by a wide margin. If flat, logging is wrong.
- [ ] Grade distribution roughly: grade 0 ≈ 85–90%, grade 4 ≈ 1–2%.
- [ ] Temporal column present (`impression_ts`) or a documented proxy for ordering sessions.

### 9.1 Regex cleaning guardrails

Applied by `shopfeed/data/clean.py` **before** the §9 checks, in the dataset build. Each rule has an action: `normalise` (rewrite in place), `drop_row`, or `drop_group` (remove the whole `qid`). Every drop is counted and written to `reports/clean_report.json`; the build fails if any single rule drops more than the threshold in the last column, because a high drop rate means the upstream log is broken, not dirty.

#### Identifier fields

| Column | Regex (full match) | Action on miss | Fail-build threshold |
|---|---|---|---|
| `session_id` | `^s_[0-9]{1,12}$` | drop_group | > 0.5% of groups |
| `qid` | `^[0-9]{1,12}$` | drop_group | > 0.1% of groups |
| `user_id` | `^u_[0-9]{1,12}$` or `^anon_[0-9a-f]{8,32}$` | drop_group | > 0.5% of groups |
| `item_id` | `^p_[0-9]{1,12}$` | drop_row | > 0.5% of rows |
| `seller_id` | `^m_[0-9]{1,12}$` | drop_row | > 0.5% of rows |

Identifiers are normalised with `strip()` and lower-casing of the prefix before matching. Anything else that does not match is treated as corruption, never coerced.

#### Categorical fields

| Column | Normalisation | Allowed set (regex) | Action on miss |
|---|---|---|---|
| `device` | `strip().lower()`; map `{"iphone","android","phone"} → mobile`, `{"pc","web","laptop"} → desktop`, `{"ios_app","android_app"} → app` | `^(mobile\|desktop\|app\|unknown)$` | drop_row |
| `user_aov_tier` | `strip().title()` | `^(Low\|Mid\|High)$` | drop_group (tier is per user, so one bad value means the user join failed) |

#### Query text

Order matters. Applied to `query_text`:

| Step | Rule | Regex | Action |
|---|---|---|---|
| 1 | Unicode NFKC normalise, `strip()`, lower-case | — | normalise |
| 2 | Collapse whitespace | `\s+` → single space | normalise |
| 3 | Strip HTML / markup | `<[^>]{1,200}>` → "" | normalise, count |
| 4 | Drop control characters | `[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]` → "" | normalise, count |
| 5 | **PII: email** | `[\w.+-]+@[\w-]+\.[\w.-]+` | drop_group, count |
| 6 | **PII: phone** | `(\+?\d[\d\s().-]{7,}\d)` | drop_group, count |
| 7 | **PII: card-like number** | `\b(?:\d[ -]?){13,19}\b` | drop_group, count |
| 8 | Injection / URL | `(https?://\|www\.\|javascript:\|data:)` | drop_group, count |
| 9 | Allowed charset after cleaning | `^[\p{L}\p{N}\p{M} '&+./#-]{1,128}$` | drop_group if miss |
| 10 | Minimum content | at least one `\p{L}` or `\p{N}` | drop_group if miss |

PII rows are dropped at the group level and the raw text is **never written** to the clean report, only the rule name and count. The original `query_text` is not retained anywhere downstream of this step.

#### Numeric fields parsed from CSV text

Numeric columns arrive as strings in CSV. Parse only if the string matches the pattern; otherwise set to NaN and count. Out-of-range values are clamped or dropped as listed.

| Column | Regex (full match) | Range rule | Action out of range |
|---|---|---|---|
| `bm25` | `^-?\d+(\.\d+)?$` | ≥ 0 | NaN |
| `semantic_sim` | `^-?\d+(\.\d+)?$` | `[-1, 1]` | NaN |
| `price` | `^\d+(\.\d{1,2})?$` (no currency symbols; strip `[$€£,]` first) | `(0, 1e6]` | drop_row |
| `hist_ctr_7d` | `^\d+(\.\d+)?$` | `[0, 1]` | NaN |
| `rating` | `^\d(\.\d+)?$` | `[0, 5]` | NaN |
| `displayed_pos` | `^[1-9]\d{0,2}$` | permutation of `1..n` within group | drop_group |
| `clicked`, `added_to_cart`, `purchased`, `wishlisted` | `^[01]$` | — | drop_row |
| `impression_ts` | `^\d{9,10}(\.\d+)?$` | within the build's data window | drop_group |
| `relevance_grade` | `^[0-4]$` | consistent with binary labels (§4.3) | drop_row |

#### Duplicate and structural rules

- Exact duplicate rows: keep first, count.
- Duplicate `(qid, item_id)` with differing labels: drop_group, count. This is a logging race, not a cleaning problem.
- Duplicate `(qid, displayed_pos)`: drop_group.
- Group left with < 2 rows after cleaning: drop_group.

#### Tests required

- Each regex has at least one positive and one negative fixture string in `tests/test_clean.py`.
- A synthetic dirty CSV (built from the 10-row fixture with one violation of every rule injected) must produce the expected `clean_report.json` counts exactly.
- The clean 10-row fixture must pass every rule with zero drops.

## 10. Risks and open questions

| # | Risk / question | Mitigation / owner |
|---|---|---|
| 1 | CSV is a 10-row fixture, not training data. | Regenerate at scale from parquet sources; §2 header is the contract. |
| 2 | Brief describes `semantic_sim` as a distance; data shows it is a similarity. | Resolved: treat as similarity. Fix the wording in the logging docs so the next person does not flip the sign. |
| 3 | No timestamp column → cannot do temporal splits or point-in-time features. | Add `impression_ts`; until then, use `session_id` ordering if monotone. |
| 4 | No randomised traffic → propensities must be estimated via EM, which is less reliable. | Ship 1% randomised-top-10 slice before the first retrain. |
| 5 | Purchase attribution window undefined. | Fix at 24h, same session; document. |
| 6 | Model learns to favour cheap items (CTR-driven). | Mean-price guardrail (§8.1), `user_aov_tier × price` cross feature. |
| 7 | Feedback loop: the ranker trains on data its predecessor produced. | Randomised slice + IPW; monitor item exposure entropy over time. |
| 8 | Purchase-dominant gain {0,1,2,5,20} is a design choice, not tuned. | Sweep the purchase gain over {10, 20, 40} on validation; pick by Purchase MRR, tie-break on NDCG@10. |
| 9 | Revenue weighting of gains. | Deferred to v2; needs a business decision on relevance vs revenue trade-off. |
| 10 | Diversity / merchant fairness re-ranking. | v2, as a post-ranking stage with its own spec. |
| 11 | Popularity features create a rich-get-richer loop; unpopular items never recover. | Learned features over hard rules, new-item exploration slot, long-tail and new-item exposure guardrails (§5.5, §1). |
| 12 | Present-day `review_score` / `popularity` in parquet leak future data. | Recompute point-in-time from events (§5.1); leakage test in §13. |
| 13 | Source log is 45 days, 100% randomised, feed not search (no `query_text`). | 60/20/20 split on 45 days ≈ 27 / 9 / 9 days. Lower §9 row/group gates for this sample and record real counts in the dev plan. Query features wait on the `search_issued` log. |
| 14 | `user_aov_tier` was read as "best value user". | Resolved: it is Average Order Value tier, a personalisation feature (§2). |

## 11. Deliverables for v1

1. `scripts/build_ranking_dataset.py`: regenerates `data/ranking_items.csv` from parquet with the §2 schema plus `impression_ts`.
2. `scripts/check_ranking_data.py`: §9 checks.
2a. `shopfeed/data/clean.py`: §9.1 regex cleaning guardrails, writes `reports/clean_report.json`.
3. `shopfeed/ranking/features.py`: feature builder incl. within-group normalisation; identical code path for train and serve.
4. `shopfeed/training/train_lambdamart.py`: §7 training, writes `artifacts/ranker_v2/`.
5. `shopfeed/eval/offline.py`: §8.1–8.4 metrics with bootstrap CIs and slices; HTML report to `reports/`.
6. Experiment plan document for the A/B in §8.5, including power calculation.
7. Tests: the full §13 test plan.
8. `shopfeed/labels.py`: event-derived grade ladder (§4.2) from `save`, `add_to_cart` and `purchase` events.
9. `shopfeed/features/history.py`: point-in-time item max-history, seller and popularity-prior features (§5.1, §5.5).
10. `shopfeed/serving/`: ranker, post-ranking rules, ranked-list cache, pagination and fallback (§5.5, §12).

## 12. Serving

### 12.1 Indexation

- **Search index** holds per item: text fields for BM25, the embedding for `semantic_sim`, and filterable fields `in_stock`, `ships_to`, `category`, `seller_id`, `price`.
- **Feature store** (key-value, keyed by `item_id` and `seller_id`) holds the §5.1 item, seller and popularity aggregates. Batch-computed daily from the event store; not in the search index, so a daily rebuild never forces a reindex.
- **Update paths**: price and stock are streamed to the index (≤ 15 min). Ratings, review counts, purchase counts and best-seller ranks are batch (≤ 24 h). New items are indexed within 15 min of `published_at` with popularity features set to the cold-start defaults, not 0.
- **Eligibility filters are applied before ranking**, in retrieval: out of stock, does not ship to the user's country, seller in the user's `blocked_merchants`, seller `status ≠ active`. The ranker never sees these items.

### 12.2 Pagination

- Rank the **whole candidate set once** (≤ 200 items) on page 1 and cache the ordered list (§12.3). Pages 2+ are slices of that list.
- Never re-rank per page. Re-ranking moves items between pages, causing duplicates and gaps; the pagination-consistency gate in §1 is 0%.
- Use an opaque cursor (`query_id` + offset), not a raw offset, so a cache miss can be detected and handled.
- On a cache miss for page 2+ (expired or evicted), re-rank, drop every item already served for this `query_id` (logged in `item_impressed`), and continue. Count these events.
- Log `page` and `page_size` in `search_issued`; `displayed_position` is the absolute position across pages (page 3, slot 2 of 20 = 42), so propensities stay comparable.

### 12.3 Caching

| Cache | Key | Value | TTL | Invalidation |
|---|---|---|---|---|
| Ranked list | `query_id` | ordered `item_id`s + model version | 30 min | none; tied to one search session |
| Query result | normalised `query_text` + `device` + `user_aov_tier` + country + filters | ordered `item_id`s | 5 min | price/stock change on any item in the list; model deploy |
| Item / seller features | `item_id`, `seller_id` | feature vector | 24 h | daily batch overwrite |
| Model | model version | loaded LightGBM booster | until deploy | blue/green swap |

The query-result cache key includes `user_aov_tier` and nothing more user-specific: it is the only personalisation input in v1, so three tiers × devices keeps the cache hit rate useful. Adding per-user features in v1.1 means dropping this cache or moving personalisation to a re-rank stage after it.

### 12.4 Latency budget (p99, page 1)

| Stage | Budget |
|---|---|
| Retrieval + filters (BM25 + ANN) | 25 ms |
| Feature fetch (batched multi-get) | 8 ms |
| Feature transform (within-group normalisation) | 2 ms |
| Model scoring, 200 candidates | 8 ms (p50 2 ms) |
| Post-ranking rules (§5.5) | 1 ms |
| Cache write + serialisation | 2 ms |
| **Ranking path total** | **≤ 21 ms** after retrieval; end-to-end ≤ 50 ms |

Pages 2+ from cache: ≤ 10 ms p99. Hard timeout on the ranking path: 30 ms, after which the fallback serves BM25 order.

### 12.5 SLA

| Item | Commitment |
|---|---|
| Availability | ≥ 99.9% monthly of search requests return a ranked list within 50 ms (≈ 43 min/month error budget). |
| Degradation | Model timeout or error → BM25 order; feature-store miss → cold-start defaults for that item; cache down → rank uncached. None of these return 5xx. |
| Fallback budget | Fallback-served responses ≤ 0.1% of requests per day; above that, page on-call. |
| Model rollback | Previous model version stays loaded; rollback ≤ 5 min, no deploy. |
| Data freshness | §12.1 update paths; a feature batch older than 48 h fires an alert and freezes retraining. |
| Error budget policy | If the monthly budget is spent, freeze model launches until the next window. |

## 13. Test plan

Every row is a test in `tests/` (or a scripted check where stated) with a binary pass criterion. CI runs unit, data-fixture and model-smoke tests on every PR; performance and full-data tests run nightly.

### 13.1 Unit tests

| Test | File | Pass criterion |
|---|---|---|
| Schema contract | `test_schema.py` | Fixture header equals §2 exactly; dtypes match. |
| Regex guardrails | `test_clean.py` | One positive and one negative string per §9.1 rule; synthetic dirty CSV yields exact expected `clean_report.json` counts. |
| Grade ladder | `test_labels.py` | Table-driven: no click → 0; click → 1; click + wishlist → 2; wishlist, no click → 2, `clicked` set to 1; ATC → 3; ATC + wishlist → 3; purchase → 4. |
| Label consistency | `test_labels.py` | Every §4.3 implication holds on the regenerated fixture. |
| Attribution | `test_labels.py` | Purchase 23 h after impression in same session attributes; 25 h or other session does not; most recent of two impressing queries wins. |
| Point-in-time features | `test_features.py` | `hist_ctr_7d`, `item_purchases_all`, `item_rating_smoothed`, `seller_*` at `t` are unchanged when events after `t` are appended. |
| Bayesian rating | `test_features.py` | Item with 1 review of 5.0 scores below an item with 500 reviews at 4.7 when category mean is 4.2. |
| Best-seller rank | `test_features.py` | Hand-built category of 5 items gets ranks 1–5 by 30-day purchases; ties broken by `item_id`. |
| Within-group features | `test_features.py` | Rank / z-score / ratio-to-max correct on a 5-row group; identical output from train and serve code paths. |
| NDCG / MRR / Purchase@k | `test_metrics.py` | Match hand-computed values for both gain tables to 1e-9. |
| Splits | `test_splits.py` | Session shares 60/20/20 ± 1 pt; no `session_id` or `qid` in two partitions; max train ts < min val ts < min test ts. |
| Post-ranking rules | `test_rules.py` | Relevance floor, unpopular demotion and new-item slot each produce the expected order on a crafted 12-item group; each flag off is a no-op. |
| Pagination | `test_pagination.py` | 200-item list paged by 20 yields 200 unique items in order; cache-miss on page 3 yields no item from pages 1–2. |

### 13.2 Data tests (scripted, `scripts/check_ranking_data.py`)

All §9 checks plus: `save` / `purchase` attribution ≥ 95%; seller and category joins with 0 orphans; feature null rate < 5%; and a 1K-sample point-in-time recomputation of every max-history feature against `events.parquet` with 0 mismatches.

### 13.3 Model tests

| Test | Pass criterion |
|---|---|
| Train smoke (`test_train_smoke.py`) | 5 trees on the fixture train without error; artefact directory has all six §7.4 files. |
| Reproducibility | Two runs with the same SHA, window and seed give identical `metrics.json` to 3 decimals. |
| Leakage | No feature > 50% of gain importance; test NDCG not more than 10% above validation NDCG. |
| Excluded features | `features.json` contains none of §5.2. |
| Directional sanity | Raising `semantic_sim`, `item_rating_smoothed` or `item_purchases_30d` for one item, all else fixed, does not lower its score on ≥ 95% of sampled groups. |
| Popularity sanity | On the test split, the model beats baseline #2a (best-seller sort) on Purchase MRR, and does not lose on the long-tail purchased-item slice (§8.2). |
| Ship gates | Every §1 technical ship gate, read from `metrics.json`. |

### 13.4 Integration and performance tests

| Test | Pass criterion |
|---|---|
| Serving parity | 1K logged requests re-scored offline and online give identical scores (train–serve skew < 0.1%). |
| Scoring benchmark (`test_serving.py`) | 200 candidates: p50 ≤ 2 ms, p99 ≤ 8 ms on serving CPU class. |
| End-to-end load | At 2× peak QPS for 30 min: p99 ≤ 50 ms page 1, ≤ 10 ms page 2+, 0 5xx. |
| Fallback | Injected model timeout, feature-store outage and cache outage each return BM25 / cold-start order, no 5xx. |
| Cache | Page-2+ hit rate ≥ 95% in the load test; price change invalidates the query-result cache entry within TTL. |
| Rollback | Switch to previous model version in ≤ 5 min under load with no failed requests. |

### 13.5 Online tests

Team-draft interleaving (3 days) then A/B per §8.5. Pass = every §1 business ship gate, guardrails green including long-tail and new-item exposure, and SLA held at each rollout stage.
