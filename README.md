# Shop feed recommendation system (interview prototype)

A runnable, tested version of the design in [shop-feed-recsys-design.md](shop-feed-recsys-design.md):
a Shop-app style home feed with retrieval, multi-task ranking, a value function, a re-ranking policy,
leakage-safe training from the impression log, offline evaluation, guardrails, A/B bucketing, and a
FastAPI serving path. Everything runs on CPU in a few minutes on synthetic data.

## Run it

```bash
pip install -e ".[dev]"
python scripts/generate_data.py          # synthetic Shop world -> data/*.parquet       (~1 s)
python scripts/train_all.py              # retrieval + light + heavy rankers -> artifacts/ (~25 s)
python scripts/evaluate.py               # offline metrics, simulated guardrails, simulated A/B (~90 s)
pytest -q                                # 98 tests                                       (~10 s)
python scripts/serve.py                  # http://127.0.0.1:8000
```

```bash
curl -s -X POST localhost:8000/feed -H 'content-type: application/json' \
  -d '{"user_id":"u_1","country":"CA","session_id":"s1","page":0}'
```

## Request path

```
FeedRequest ─► build_context ─► generate_candidates ─► light ranker ─► heavy ranker ─► value ─► rerank ─► log
                 (streaming        6 capped sources,      LR, cut        MMoE-lite,      Σ w·p      diversity,   one Impression
                  store, follows,  eligibility applied    2k→300        7 heads         scaled by   freshness,   per card with
                  blocked, hidden) BEFORE ranking                                       order value exploration, features, scores,
                                                                                                    merchant cap propensity, version
```

| Stage | Module | Notes |
|---|---|---|
| Context | `serving/pipeline.py::build_context` | repeat-block set, hidden set, followed, purchase history, user embedding (prior vector for cold users) |
| Candidates | `retrieval/sources.py` | two-tower ANN, followed merchants, complements, replenishables, fresh lane, trending; eligibility is a pure mask applied first |
| Light ranker | `ranking/light.py` | logistic regression on 13 cheap features |
| Heavy ranker | `ranking/heavy.py` | shared bottom, 3 experts, per-head gates, heads: click, dwell, save, atc, purchase, hide, return |
| Value | `ranking/value.py` | purchase/atc scaled by clipped order value, hide and return negative; weights in `config.py` |
| Re-rank | `rerank/policy.py` | ≤2 consecutive same merchant, ≤3 same category in top 10, ≥2 fresh items, 2 exploration slots with logged propensity, merchant share cap 30% |
| Logging | `serving/logger.py` | every card: features, head scores, value, position, propensity, model version |

## Features and leakage

`features/snapshot.py` is the single place a feature vector is assembled, for training and serving.
Batch features (`features/batch.py`) are computed from events strictly before an `as_of` time and are
refreshed daily; the training set rebuilds them per day bucket so no row sees its own future.
Streaming counters (`features/streaming.py`) take an explicit clock and are queried at each
impression's own timestamp. Position is the last feature: trained on, zeroed at inference.

Training examples (`training/dataset.py`): one per logged product impression whose 7-day label window
has closed; labels come only from events linked by `impression_id` inside the window; negatives are
shown-but-unengaged impressions; the validation split is time-ordered with a 7-day gap.

## Cold start

* **New product**: the item tower uses catalog content only, so `FeedService.add_products` embeds and
  indexes a product at publish time. It is also eligible for the fresh lane and the freshness floor.
* **New user**: a prior user embedding (uniform affinity, empty profile) plus followed/fresh/trending
  sources. Shop Pay purchases count as history, so a Shop-Pay-only user is personalised from the start.

## Results on the default synthetic world (2000 users, 4000 products, 196k logged impressions)

| | logging policy | model (simulated) |
|---|---|---|
| click rate | 0.118 | 0.211 |
| purchase rate | 0.017 | 0.035 |
| hide rate | 0.013 | 0.007 |

Validation AUC: purchase 0.73, save 0.72, atc 0.71, hide 0.64. p95 latency 89 ms against a 150 ms budget.
Simulated A/B (model vs random policy over eligible candidates): +42 GMV per user, 95% CI [11, 83].
All four guardrails pass for the model arm. Numbers come from `scripts/evaluate.py`; the simulator
replays the generator's own engagement model, so they demonstrate the mechanics, not real-user lift.

## Tests

`tests/` covers unit behaviour per module, edge cases (cold user, Shop-Pay-only user, cold product,
no eligible products, duplicate and future events, NaN features, empty log, candidate pool smaller
than the page) and business cases (complements not substitutes, replenish windows, followed drops,
hidden items never return, repeat block, merchant cap on every page, exploration propensities,
clickbait ablation, model beats random on simulated GMV). Run `pytest -q`.

## What is deliberately simplified

Brute-force cosine index instead of FAISS; pandas instead of a feature store; a synthetic world with a
known engagement formula; no categorical embeddings for user/product ids in the ranker; batch refresh
is a method call, not a job. Each has the interface a production component would expose.
