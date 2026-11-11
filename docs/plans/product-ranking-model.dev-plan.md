# Development Plan: Product Ranking Model (LambdaMART v1)

| | |
|---|---|
| Status | Draft v1 |
| Owner | Alex Lavrenin |
| Date | 2026-11-09 |
| Spec | [`docs/specs/product-ranking-model.md`](../specs/product-ranking-model.md) |
| Target artefact | `artifacts/ranker_v2/` |

This plan turns the spec into an ordered sequence of milestones. The spec says *what*; this document says *in what order, in which files, and when each step is done*. Section references (§) point to the spec.

## 0. Principles

- **Spec is the contract.** Every milestone cites the spec section it satisfies. If implementation forces a deviation, update the spec in the same PR.
- **Evaluation before model.** The offline harness (M3) lands before any ranker is trained. A ranker without trusted NDCG and Purchase MRR numbers cannot be reviewed.
- **Fixture-first.** Every module must run end-to-end on the 10-row `data/ranking_items.csv` before it touches the full dataset. Tests use the fixture.
- **Purchase is the objective.** Any metric or loss change is judged by Purchase MRR first, NDCG@10 second (§1, §4.2).
- **One PR per milestone.** Conventional commits. Tests required for every new module. `ruff` + `pytest` green before push.

## 1. Starting point

- Git tracks only `.gitignore` and `LICENSE`. The `shopfeed/` package exists as empty directories with stale `__pycache__`; the `egg-info/SOURCES.txt` lists modules that no longer exist. Treat the codebase as **greenfield**.
- `data/` holds `impressions.parquet`, `events.parquet`, `products.parquet`, `users.parquet`, `merchants.parquet`, `follows.parquet` and the 10-row CSV fixture.
- `artifacts/ranker_v1/metrics.json` records NDCG@20 = 0.440 from a prior torch model. Use as a historical baseline only; do not attempt to reproduce it.
- System Python is 3.9.6 with no `lightgbm` or `pandas`. The project pins Python ≥ 3.10 in a virtualenv.

## 2. Milestones

Sequential. Each is sized for roughly one engineer-week. Exit criteria are binary.

| # | Milestone | Spec § | Exit criterion |
|---|---|---|---|
| **M0** | Project skeleton | — | `pyproject.toml` with `lightgbm`, `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `pytest`, `ruff`; `shopfeed/` modules stubbed with docstrings; `tests/test_schema.py` passes on the fixture; CI runs `ruff check` + `pytest`. Randomised-top-10 traffic request filed (§3.4). |
| **M1** | Dataset build + cleaning + readiness checks | §2, §3, §9, §9.1 | `scripts/build_ranking_dataset.py` regenerates the CSV from parquet with the §2 header **plus `impression_ts`**. `shopfeed/data/clean.py` applies the §9.1 regex guardrails (ID formats, categorical normalisation, query-text PII and injection drops, numeric parsing, duplicates) and writes `reports/clean_report.json`; build fails if any rule exceeds its drop threshold. `scripts/check_ranking_data.py` implements every §9 check, passes on the full set, and fails on the fixture by design (row-count gate). Real row/group counts and clean-report totals recorded in this doc (§6 below). |
| **M2** | Labels + feature builder | §4, §5.1–5.3 | `shopfeed/labels.py` derives `relevance_grade` from events with 24 h same-session attribution and enforces the §4.3 consistency rules. `shopfeed/features/ranking.py` builds all v1 feature groups including within-group rank / z-score / ratio-to-max, with one code path for train and serve. Point-in-time `hist_ctr_7d` test green. |
| **M3** | Offline evaluation harness | §8.1–8.4 | `shopfeed/eval/metrics.py`: `ndcg_at_k` (both gain tables), `purchase_mrr`, `purchase_at_k`. `shopfeed/eval/offline.py`: bootstrap CIs over `qid`, paired deltas vs production order, §8.2 slices, HTML report to `reports/`. NDCG verified against a hand-computed example. |
| **M4** | Baselines | §6.1 rows 0–3 | Logged order, BM25 sort, CTR sort and pointwise LightGBM scored through M3. Numbers frozen in `reports/baselines.json` with CIs. |
| **M5** | LambdaMART v1 | §6.2, §7 | `shopfeed/training/splits.py`: temporal 60 / 20 / 20 train / validation / test split by session start time, session-level, cut timestamps written to `meta.json`, user-seen/unseen flag; `tests/test_splits.py` asserts shares within ±1 pt, no group straddles a partition, and max train time < min validation time < min test time. `shopfeed/training/train_lambdamart.py` with §7.3 config writes `artifacts/ranker_v2/`: `model.txt`, `features.json`, `propensities.json`, `metrics.json`, `meta.json`, `importance.csv`. Beats logged order by ≥ 2% NDCG@10 on test with lower CI > 0. Feature-importance leakage check (§5.3) documented. |
| **M6** | Position debiasing | §5.4 | `shopfeed/eval/propensity.py`: EM-based propensity curve from logs, with a code path for the randomised slice when it exists. `lambdarank_unbiased` enabled. IPW-NDCG reported alongside raw NDCG. Propensities clipped at 0.1 and versioned with the model. |
| **M7** | Gain sweep + hyperparameter search | §4.2 risk 8, §7.3 | `shopfeed/training/sweep.py`: purchase gain ∈ {10, 20, 40} × ~30 random-search trials on validation. Selection by Purchase MRR, tie-break NDCG@10. Chosen config committed as the default in `train_lambdamart.py`. |
| **M8** | Serving + fallback | §1 technical metrics | `shopfeed/serving/ranker.py`: loads artefact, scores ≤ 200 candidates in ≤ 2 ms p50 / ≤ 8 ms p99 on CPU, falls back to BM25 order on any error, logs features per request for train–serve skew measurement. Benchmark test included. |
| **M9** | Experiment plan + launch readiness | §8.5, §8.6 | `docs/experiments/ranker-v2-ab.md`: power calculation for +1% relative purchase conversion, 3-day interleaving pre-check, metric and guardrail definitions, rollout ladder 1→10→50→100%, 4-week 5% holdback. Baselines marked TBD in spec §1 filled in from holdback week 1. |

## 3. Module map

```
pyproject.toml
shopfeed/
  __init__.py
  config.py                    paths, seed, data window, attribution window
  schema.py                    CSV column contract (§2): dtype dict + validate()
  labels.py                    relevance_grade from events, §4.3 rules        (M2)
  data/
    build.py                   parquet → ranking_items.csv                    (M1)
    clean.py                   §9.1 regex cleaning guardrails + clean report  (M1)
    checks.py                  §9 readiness checks                            (M1)
  features/
    ranking.py                 feature groups + within-group normalisation    (M2)
  eval/
    metrics.py                 ndcg_at_k, purchase_mrr, purchase_at_k         (M3)
    offline.py                 bootstrap, paired deltas, slices, HTML report  (M3)
    propensity.py              EM / randomised-slice propensity curve         (M6)
  training/
    splits.py                  temporal session-level split                   (M5)
    train_lambdamart.py        §7 training, writes artifacts/ranker_v2/       (M5)
    sweep.py                   gain + hyperparameter search                   (M7)
  serving/
    ranker.py                  load, score, fallback, request logging         (M8)
scripts/
  build_ranking_dataset.py     CLI over data/build.py
  check_ranking_data.py        CLI over data/checks.py
  train.py  evaluate.py  sweep.py
tests/
  test_schema.py  test_clean.py  test_labels.py  test_features.py  test_metrics.py
  test_splits.py  test_train_smoke.py (fixture, 5 trees)  test_serving.py
docs/experiments/ranker-v2-ab.md                                              (M9)
```

## 4. Dependencies and blockers

Carried over from spec §10 and the repo audit.

| Blocker | Blocks | Action |
|---|---|---|
| `impression_ts` absent from CSV | M1, M5 | Build script derives it from `impressions.parquet`. If the parquet has no timestamp, **stop and raise**; temporal splits are not negotiable. |
| Purchase attribution window undefined | M2 | Fixed at 24 h, same session. Documented in `labels.py` docstring and `config.py`. |
| No randomised traffic | M6 | Ship EM propensities first. Randomised-top-10 request filed at M0 so data exists by the first retrain. |
| System Python 3.9 | M0 | Project requires ≥ 3.10 in a venv. Never run against system Python. |
| Dataset scale unknown | M1 | §9 thresholds (≥ 1M rows, ≥ 50K groups) are set for production logs. If the parquet is a synthetic sample, lower them in the same PR and record the real counts in §6. |
| Wishlist / ATC events may be missing from parquet | M2 | If so, grade ladder collapses to {0, 1, 4}. Gain table still valid; Purchase MRR becomes the only informative ranking metric. Flag in spec. |

## 5. Risks to this plan

- **Scope creep into neural ranker.** Spec §6.2 defers it. Do not start it before M9 closes, regardless of how M5 numbers look.
- **Tuning on test.** M7 runs on validation only. The test split is read once, at the end of M7, and the number is frozen.
- **Over-aggressive cleaning.** §9.1 thresholds fail the build on high drop rates so a bad regex cannot silently shrink the dataset. If a rule trips, inspect the counts before loosening it.
- **Train–serve drift.** M2 builds one feature code path. If serving later needs a different implementation for latency, add a parity test before merging it.
- **Recovering `ranker_v1`.** Sources are gone. Do not spend time on it.

## 6. Dataset facts (fill in at M1)

| Fact | Value |
|---|---|
| Rows | TBD |
| Query groups | TBD |
| Sessions / users / items | TBD |
| Date range | TBD |
| Grade distribution 0 / 1 / 2 / 3 / 4 | TBD |
| CTR at position 1 vs 10 | TBD |
| Median candidates per group | TBD |
| Rows / groups dropped by §9.1 cleaning, by rule | TBD |

## 7. Definition of done for v1

- M0–M9 merged to `main`.
- `scripts/check_ranking_data.py` green on the full dataset.
- `artifacts/ranker_v2/metrics.json` meets every **technical ship gate** in spec §1: NDCG@10 ≥ +2% on test and on the randomised slice, Purchase MRR ≥ +3%, Purchase@10 ≥ production, scoring latency within budget.
- `docs/experiments/ranker-v2-ab.md` approved.
- Business ship gates (spec §1) are decided by the A/B, not by this plan.

## 8. Traceability to spec deliverables (§11)

| Spec deliverable | Milestone |
|---|---|
| 1. `scripts/build_ranking_dataset.py` | M1 |
| 2. `scripts/check_ranking_data.py` | M1 |
| 2a. `shopfeed/data/clean.py` regex guardrails | M1 |
| 3. Feature builder | M2 |
| 4. `train_lambdamart.py` → `artifacts/ranker_v2/` | M5, M6, M7 |
| 5. Offline eval with CIs, slices, HTML report | M3 |
| 6. Experiment plan with power calculation | M9 |
| 7. Tests: labels, leakage, split integrity, NDCG | M2, M3, M5 |
