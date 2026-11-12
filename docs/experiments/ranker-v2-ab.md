# Experiment plan: ranker_v2 (LambdaMART v1) vs production order

| | |
|---|---|
| Status | Draft for review |
| Owner | Alex Lavrenin |
| Date | 2026-11-11 |
| Spec | [`docs/specs/product-ranking-model.md`](../specs/product-ranking-model.md) §1, §8.5, §8.6 |
| Candidate | `artifacts/ranker_v2/` (M7 config: one 7-leaf tree, purchase gain 40, position-debiased) |
| Control | Current production ordering |

## 1. Hypothesis

Ranking search candidates with ranker_v2 increases **purchase conversion per search session** compared with the production order. It must not hurt the guardrails: reformulation, zero-click, latency, returns, merchant concentration, price mix, long-tail and new-item exposure.

Offline evidence (test split, the randomised 45-day feed sample; dev plan §6a):

| | NDCG@10 | IPW-NDCG@10 | Purchase MRR | Purchase@10 |
|---|---|---|---|---|
| logged (random) order | 0.356 | 0.315 | 0.287 | 0.626 |
| semantic-sim sort | 0.519 | 0.512 | 0.520 | 0.927 |
| ranker_v2 | 0.535 | 0.517 | 0.529 | 0.935 |

Caveats that cap how much to trust these numbers:
- The offline reference is a *random* order, not production. Production already applies some relevance ordering, so the online lift will be far smaller than +50% NDCG.
- `semantic_sim` is a user–item proxy and carries most of the signal. The data is synthetic feed traffic, not search.
- Against the strongest simple baseline (semantic sort), only NDCG@10 is significantly better (+3.0%, CI +1.6 to +4.7). Purchase MRR is +1.7%, n.s.

**Do not launch the A/B on the sample-trained model.** Retrain on production search logs that have `query_text`, real `bm25` and query–item `semantic_sim`. Rerun M4–M7, then come back to this plan.

## 2. Design

- **Unit**: user-level randomisation, hash(user_id, salt) → arm. It is sticky across sessions so carry-over cannot contaminate the control.
- **Arms**: control = production order; treatment = ranker_v2 with the §5.5 rules (relevance floor, unpopular demotion, new-item slot on; best-seller boost off).
- **Held fixed**: retrieval, eligibility filters, page size, UI. Only the order of the candidate set changes.
- **Duration**: at least 14 days (two weekly cycles) and until the sample size in §4 is reached, whichever is longer.
- **Exposure logging**: every `search_issued` and `item_impressed` event carries `arm` and `model_version`.

### 2.1 Interleaving pre-check (3 days)

Run team-draft interleaving of ranker_v2 against production on 5% of traffic for 3 days before the A/B.
- **Pass**: ranker_v2 wins more purchase-attributed credits than production, with a binomial test p < 0.05.
- **Abort**: ranker_v2 loses on clicks *and* purchases. That means the model or the feature pipeline is broken; do not start the A/B.

Interleaving needs about 10× less traffic than an A/B for ranking comparisons (spec §8.5).

## 3. Metrics and decision rules

Definitions follow spec §1. Every ratio metric is per search session and attributed per §3.1: a backend-stamped `query_id` is trusted up to 7 days; otherwise reconstruct within 24 h.

| Metric | Role | Decision rule |
|---|---|---|
| Purchase conversion per search session | **Primary** | Ship only if treatment ≥ control and the lower bound of the 95% CI of the relative lift > 0 |
| Revenue per search session | Primary (business) | Lower bound of the 95% CI ≥ 0 |
| ATC rate per search session, wishlist rate, CTR@10 | Secondary | Informational; expect a lift |
| Best-seller coverage in top 10 | Secondary | ≥ control |
| Query reformulation rate (2nd query < 60 s, no click) | Guardrail | Relative increase ≤ 2% |
| Zero-click rate | Guardrail | Relative increase ≤ 2% |
| Search latency p99 | Guardrail | ≤ control + 10 ms |
| Fallback-served responses | Guardrail | ≤ 0.1% of requests per day |
| Mean price of top 10 | Guardrail | Within ±5% of control |
| Merchant GMV share of top 1% merchants | Guardrail | Increase ≤ 1 pt |
| Long-tail exposure, new-item exposure (top 10) | Guardrail | Relative drop ≤ 10% |
| 30-day return rate on search-attributed orders | Guardrail (delayed) | No increase; read in the holdback |

**Statistics**
- Delta method for ratio metrics, CUPED on the user's pre-period purchase rate.
- Two-sided α = 0.05 for the primary metric. Guardrails are one-sided, Bonferroni-corrected across the 9 guardrails (α = 0.0056 each).
- No peeking: fixed-horizon analysis at the planned sample size. The daily guardrail dashboard is only for safety stops (§5).

## 4. Power calculation

Two-proportion test, α = 0.05 two-sided, power = 0.80, n per arm = (z₀.₉₇₅ + z₀.₈)² · (p₁(1 − p₁) + p₂(1 − p₂)) / (p₂ − p₁)².

The baseline comes from the 45-day sample: p₁ = 0.279 purchase conversion per search session, 217 sessions/day. **Replace p₁ and traffic with production numbers from holdback week 1 before scheduling.**

| Relative lift | Sessions per arm | Total sessions | Days at sample traffic |
|---|---|---|---|
| +1% (spec sizing) | 406,524 | 813,049 | 3,739 |
| +3% (ship gate) | 45,438 | 90,877 | 418 |
| +5% | 16,452 | 32,904 | 151 |
| +10% | 4,169 | 8,338 | 38 |

Revenue per session (mean 27.9, sd 60.9 on the sample) needs 749,605 sessions per arm for +1% and 83,289 for +3%. Revenue therefore drives the sample size if it is a ship gate.

**Implications**
- The +1% sizing in spec §8.5 needs about 0.8M search sessions, which is only feasible on production traffic. At sample traffic, only lifts ≥ 10% are detectable in a reasonable time.
- CUPED typically cuts the required n by 20–40% when pre-period purchase behaviour is predictive. Measure that variance reduction on holdback-week-1 data before relying on it.
- If production traffic cannot reach the +1% sizing in 4 weeks, size for the +3% ship gate and say so in the readout. Treat a smaller true lift as undetectable, not as zero.

## 5. Rollout and safety

| Stage | Traffic to treatment | Minimum duration | Advance when |
|---|---|---|---|
| 0 | Interleaving 5% | 3 days | §2.1 pass |
| 1 | 1% | 3 days | All guardrails green, fallback ≤ 0.1%, no latency regression |
| 2 | 10% | 3 days | Same |
| 3 | 50% (A/B readout arm) | ≥ 14 days and n reached | Primary decision rule met |
| 4 | 100% with 5% holdback | 4 weeks | Holdback confirms the lift; 30-day return rate not up |

**Automatic stop**, with rollback to the previous model version in ≤ 5 min, if any of these hold at any stage:
- p99 search latency exceeds control + 10 ms for 30 min;
- fallback rate > 1% for 15 min;
- zero-click or reformulation rate is up > 5% relative on a daily read.

## 6. Readout

- A primary metric table with CI and p-value.
- Every guardrail with its CI.
- Slices from spec §8.2: device, AOV tier, new vs returning, query length, candidate-set size, top categories, seller size, best-seller vs long-tail purchases.
- The offline–online consistency point for spec §8.6: offline NDCG@10 delta vs online conversion delta, logged so the correlation can be tracked across experiments.
- A ship / no-ship decision against spec §1 business gates. "Not significant" is reported as "no detectable effect at this power", not as "no effect".

## 7. Open items before launch

1. Retrain on production search logs (query text, BM25, query–item similarity). This plan's offline numbers come from a synthetic feed.
2. Replace the per-request feature computation with a precomputed key-value table. Feature time is about 18 ms and over the §12.4 budget.
3. Freeze production baselines (p₁, revenue, guardrail levels) from holdback week 1 and recompute §4.
4. Confirm the attribution rule (trusted stamp up to 7 days) with analytics so both arms are measured the same way.
