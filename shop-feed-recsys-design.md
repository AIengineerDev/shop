# Shop App Feed Recommendation System — Interview Design

## 0. Clarifying assumptions (stated up front)

- Surface: the Shop app home feed. Mixed item types: product cards, merchant/brand cards,
  collections ("drops"), and order-status cards for in-flight orders.
- Product objective: GMV through Shop (incremental, not cannibalized from merchant storefronts),
  with 28-day buyer retention as the long-horizon metric. Session-level proxy: a "useful session"
  has >= 1 product detail view with meaningful dwell OR a save/follow/add-to-cart, and no
  hide/not-interested on the first screen.
- Eligibility: merchant active and in good standing, product in stock in the user's region,
  ships to the user's country, price in local currency, not already purchased or shown in the
  last 7 days, passes trust & safety. Freshness: new drops from followed merchants must be
  surfaceable within minutes.
- Signals available (Shopify-specific advantage): full order history across ALL Shop merchants
  (not just one store), Shop Pay purchase graph, merchant follows, saves/wishlists, product
  views, add-to-cart, checkout abandon, hide / not interested, delivery and return events,
  merchant catalog metadata (title, description, images, category, price, inventory, variants).
- Baseline: followed-merchant chronological feed + category popularity backfill.

## 1. Serving architecture (budget ~150 ms p99)

Request -> [Candidate generation, parallel sources, ~2k items]
        -> [Light ranker on cheap features, ~300]
        -> [Heavy multi-task ranker]
        -> [Re-rank: diversity, freshness, exploration, business rules]
        -> page of ~20 cards, logged with features + scores + position + model version

Candidate sources (each capped so none dominates):
1. Two-tower ANN retrieval (user embedding vs product embedding).
2. Followed merchants' recent products / drops.
3. Item-to-item: complements of recent purchases (bought shoes -> socks, not more shoes).
4. Replenishables: consumables with a predicted repurchase window (coffee, skincare).
5. Fresh/new merchants and products with content-only embeddings (cold start lane).
6. Regional trending (velocity of orders in the user's country over the last few hours).

Feature stores:
- Batch (daily): long-term category affinity, price-band affinity, merchant quality
  (fulfilment speed, return rate, review score), LTV segment.
- Streaming (seconds): item orders/views last hour, current-session actions, inventory.
- Embeddings: product vector computed at publish from catalog content; user vector updated
  near-real-time from recent actions.
- Every feature is logged at request time next to the impression -> no train/serve skew.

## 2. Modeling

Retrieval: two-tower, in-batch sampled softmax with logQ popularity correction.
Positives = meaningful engagements (detail view with dwell, save, ATC, purchase), weighted by
strength. Item tower is content-based (text + image embeddings, category, price band,
merchant id) so brand-new products get a usable vector before any interaction.

Ranking: multi-task network (shared bottom, MMoE-style gating), one head per action:
P(click), P(dwell>threshold), P(save), P(add-to-cart), P(purchase | shown), P(hide),
P(return | purchase). Binary cross-entropy per head, prevalence-weighted.
Final value = sum_k w_k * p_k, where weights encode the product objective:
purchase and ATC weighted by expected margin-adjusted order value, hide and return strongly
negative. Weights tuned against retention/GMV in online tests, not clicks.
Why separate heads: Shopify can change what it values (e.g. favor new merchants, penalize
high-return categories) without retraining.

Training data & leakage:
- One example per logged impression with the serving-time feature snapshot.
- Labels attached after a fixed 7-day window (purchase decisions are slow), same window for all.
- Ranking negatives = shown-but-not-engaged impressions, never unseen items.
- Position fed as a feature in training, fixed to slot 1 at inference (position-bias tower).
- Strict time-ordered splits. Order-status cards excluded from training labels (they are
  utility, not recommendations).

Feedback bias & exploration:
- A small share of slots is randomized from the retrieval pool with logged propensity;
  IPS-weighted when training and when estimating counterfactual value offline.
- Cold-start boost on new products/merchants that decays as prediction confidence rises.
- Merchant concentration cap: top 1% of merchants cannot exceed X% of impressions.

## 3. Evaluation

Offline (gate only): per-head AUC / logloss, NDCG@20 on value score, calibration of
P(purchase) by decile (needed because value score multiplies by order value).
Online (decides ship): A/B on GMV per user and 28-day buyer retention, with guardrails:
hide / not-interested rate, return rate, merchant concentration, fresh-item share,
cannibalization check against merchants' own storefront orders. Long-term holdback group
on the old model for 4+ weeks because purchase effects lag.

## 4. Follow-ups

New users / new items: new users seeded from Shop Pay history if they have ever checked out
with Shop Pay on any store (huge Shopify advantage), otherwise onboarding category picks +
regional popularity, with in-session updates from first dwell signals. New products enter
via the content-embedding lane with a decaying boost, shown first to a small diverse audience;
early conversion vs that audience's baseline decides graduation into the main ANN index.

Engagement up, hides up: value function is wrong (model found provocative/clickbait
listings). Hide rate is a hard ship guardrail; increase hide weight and retrain; check whether
GMV/retention actually moved (usually flat or down with lag); inspect which categories drive
both signals (often one polarizing vertical or a dropshipping cluster); check if hides
concentrate in exploration slots, which is a tuning problem not a ranking problem.
