# Development Plan: Catalog Onboarding & Taxonomy Normalization v1 completion

| | |
|---|---|
| Status | In progress |
| Owner | Alex Lavrenin |
| Date | 2026-11-12 |
| Spec | [`docs/specs/catalog-onboarding-taxonomy.spec.md`](../specs/catalog-onboarding-taxonomy.spec.md) |
| Base | branch `taxonomy` (8 commits: scaffold → pipeline → API → e2e) |
| Work branch | `task/catalog-v1-completion` |

The `taxonomy` branch already implements most of the spec. This plan closes the gaps found by a criterion-by-criterion audit and adds the same CI test layers as the ranking project: lint, types, unit, integration, e2e, coverage, load and security. Section references (§, FR, AC) point to the spec.

## 1. Starting point (audited 2026-11-12)

- **Green:** 105 tests (+1 `slow`), `ruff` clean, `mypy --strict` clean, 94% line coverage. pip-audit and gitleaks are clean.
- **About 30 of 40 spec items are implemented and tested:** FR-1 rules 1–5 and 8, AC-1.1–1.3, AC-1.6, AC-2.1, AC-2.3, AC-2.6, the FR-2c table, FR-3, AC-3.1–3.7, all nine FR-4 endpoints, AC-4.2–4.5.
- **No CI workflow** on the branch.

### Defects reproduced before planning

| # | Defect | Reproduction | Spec |
|---|---|---|---|
| D1 | A non-UTF8 byte in any row aborts the **whole batch** (exit 1, no summary) | `raw_title = x'…ff fe…'` → `sqlite3.OperationalError` in `store.iter_feed_rows`, outside the per-item try | AC-1.5, AC-3.5, AC-3.6 |
| D2 | Thousands separators break prices | `"$1,299.99"` → range 1.29–9.99; `"USD 1,000"` → min 0, so a valid listing is quarantined `PRICE_INVALID` | FR-1 rule 7 |
| D3 | `Nx` pack form not recognised | `"Crew Socks 3x"` → `pack_count = None` | FR-1 rule 6 |
| D4 | bandit **High**: SHA1 for the duplicate fingerprint; `assert` as runtime guards (stripped under `-O`) | `comply.py:35`, `engine.py:350`, `validate.py:89` | §8 |

### Gaps

| # | Gap | Spec | Size |
|---|---|---|---|
| G1 | No concurrency: items are processed sequentially; no throughput test | §8 (≥ 200 items/min at concurrency 8) | L |
| G2 | No re-validation from stored candidates; taxonomy not in `rules_version`; drafts don't store the candidate | AC-2.4 | M |
| G3 | Log lines never carry `feed_item_id` / `run_id`; per-item output uses `print`; API has no JSON logging | §8 | M |
| G4 | No `.env` support; default model not Sonnet-class; no test for `Settings.from_env` | §8 | S |
| G5 | `LANG_UNSUPPORTED` never emitted | §10 | S |
| G6 | `Any`-typed fields in API models (`dict[str, Any]` on trace / raw / extra) | AC-4.1 | M |
| G7 | Reason codes drift: `LLM_SCHEMA_INVALID:classify` vs spec's bare code; `CLASSIFICATION_OUT_OF_SET` can never fire; `agree` set without checking the threshold | AC-2.2, FR-2a | S |
| G8 | 100% branch coverage of `validate.py` claimed but not measured; untested branches | AC-2.5 | S |
| G9 | AC-4.6 measured only for one filtered draft list; `q` / `reason_code` filters untimed; `slow` marker not skipped | AC-4.6 | S |
| G10 | Test hygiene: snapshot test writes its own golden file when missing; no hypothesis test for the word-boundary matcher; no network guard; no ≤ 2 LLM-calls-per-item assertion; AC-4.5 not checked through the API for a quarantined item | §9, AC-1.4, AC-3.3, §8 | S |

## 2. Milestones

One commit per milestone on the work branch; one PR into `taxonomy`. Every milestone adds tests first (red), then the fix (green).

| # | Milestone | Closes | Exit criterion |
|---|---|---|---|
| **C0** | Security fixes + test infrastructure | D4, G8 (measurement), G9 (marker), G10 (guards) | bandit 0 findings at all severities; pytest-cov with branch coverage; markers `integration` / `e2e` / `load` registered and excluded from the default run; socket guard fails any network call in tests; snapshot test fails on a missing golden file |
| **C1** | Robust ingestion | D1, D2, D3, G5 | Non-UTF8 row → that row is quarantined (`TEXT_ENCODING_INVALID` + `PIPELINE_ERROR:read` if still unreadable), batch completes, exit 0; prices with thousands separators (`1,299.99`, `1.299,99`, `USD 1,000`) parse; `3x` / `3X` / `x3` packs parse without matching `2x4`; non-ASCII-script titles get `LANG_UNSUPPORTED` (warning) |
| **C2** | Classification and reason-code correctness | G7 | `classification_source = "agree"` only when top-1 ≥ `HIGH_CONF_THRESHOLD`; out-of-set id → `CLASSIFICATION_OUT_OF_SET` (blocking) after retries; schema failures → `LLM_SCHEMA_INVALID` with the stage in `detail`; ≤ 2 LLM calls per item asserted; spec §11 answers written back into the spec |
| **C3** | Re-validation without the LLM | G2 | Drafts store `candidate_json`; `python -m catalog_onboard revalidate --db …` re-runs validate → comply → route from stored candidates with **0 LLM calls**; taxonomy content hashed into `rules_version`; changing a category's `required_attributes_json` moves affected items between draft and quarantine |
| **C4** | Observability and config | G3, G4 | JSON log lines with `feed_item_id` and `run_id` on every per-item line (contextvar); API logs JSON; `.env` loaded (env vars win); default model Sonnet-class; tests for both |
| **C5** | Concurrency and throughput | G1 | Classify / extract stages run in a bounded pool (`--concurrency`, default 8); a single writer thread owns SQLite; outcomes identical to the sequential run; ≥ 200 items/min with a fake LLM at realistic latency |
| **C6** | API typing and latency | G6, G9 | No `Any` in the OpenAPI schema (`additionalProperties: true` rejected by a test); p95 < 100 ms on 10k rows for **every** list endpoint and filter, including `q` and `reason_code` |
| **C7** | CI | all | Jobs: lint, types, unit (coverage gate), integration, e2e, coverage (unit + integration + e2e including CLI subprocesses), load, security. All green on GitHub |

## 3. Test layers (same as the ranking project, adapted)

| Layer | Marker / job | What it covers here |
|---|---|---|
| Unit | default `pytest` | cleaning rules, extractors, each validation branch (100% branch on `validate.py`), compliance matcher (hypothesis), LLM schema contract |
| Integration | `-m integration` | starter DB → engine (fake LLM) → store → read-only API, in-process: routing invariants, supersede, revalidate, trace content |
| E2E | `-m e2e` | `make_starter_db.py` → `python -m catalog_onboard ingest` (rerun = no-op) → `revalidate` → `scripts/serve.py` as a **real HTTP server** queried over the network stack |
| Coverage | `-m "not load"` | unit + integration + e2e with CLI subprocesses counted (coverage `patch = ["subprocess"]`) |
| Load | `-m load` | AC-4.6: p95 < 100 ms per list endpoint and filter on 10k rows; §8: ≥ 200 items/min ingest with concurrency 8 against a fake LLM with injected latency; concurrent API clients |
| Security | `security` job | pip-audit (project deps), bandit (all severities, justified `nosec` only), gitleaks (full history) |
| Types / lint | `types`, `lint` | `mypy --strict catalog_onboard`, `ruff check .` |

## 4. Risks

- **Concurrency with SQLite.** Workers must not write. Only the LLM-bound stages run in parallel; persistence stays on one thread in input order, so outcomes and `pipeline_runs` order stay deterministic.
- **Changing `rules_version` inputs (taxonomy hash) changes every version string.** Existing DBs will re-process on the next ingest (supersede, not duplicate). This is the intended AC-3.4 behaviour; call it out in the PR.
- **Reason-code renames** (`LLM_SCHEMA_INVALID:<stage>` → bare code + detail) change `/quarantine?reason_code=` filters. The API filter will accept both forms for one release.

## 5. Results log

| Milestone | Outcome |
|---|---|
| C0 | bandit 0 findings (SHA1 `usedforsecurity=False`, asserts → explicit checks); network guard; snapshot no longer self-heals; `validate.py` 88% → **100% branch** |
| C1 | One non-UTF8 row no longer aborts the batch (was exit 1, no output). `$1,299.99` no longer parses as a 1.29–9.99 range; `USD 1,000` no longer parses as 0 (false `PRICE_INVALID`). `3x` packs parsed; `2x4` / `3 x 4` / `3XL` are not. `LANG_UNSUPPORTED` emitted |
| C2 | Distinct `LLM_SCHEMA_INVALID` / `CLASSIFICATION_OUT_OF_SET` (was unreachable) / `LLM_TRANSPORT_ERROR`; extract failure blocks; `agree` respects the threshold; ≤ 2 LLM calls/item asserted |
| C3 | `revalidate` CLI: 0 LLM calls, idempotent per version; version = rules + taxonomy content; drafts store their candidate |
| C4 | `feed_item_id` / `run_id` on every JSON log line; API access log in every worker process; `.env`; defaults `claude-sonnet-5-5`, effort `low`, `max_tokens` 4096 |
| C5 | Concurrency 8: outcomes, duplicate detection and trace order identical to sequential; **837 items/min** at 0.5 s/call (target ≥ 200) |
| C6 | No `Any` in OpenAPI (a typed recursive `JsonData` schema; the old test missed `additionalProperties: true`) |
| C7 | Test layers + CI. The load test found three serving defects, all fixed: wrong-thread SQLite connection (500s under overlap), GIL convoy in concurrent sqlite3 reads (500-row pages 1,064/s → 36/s on 8 threads), and worker processes emitting no logs. Also fixed: `close()` racing an in-flight statement (an interpreter crash in testing), and invalid env values exiting with a traceback instead of exit 2 |

### Final measurements

| Check | Result |
|---|---|
| Unit / unit+integration+e2e branch coverage | 94% / 94% (gates 92% / 93%); `validate.py` 100% |
| AC-4.6 worst p95, 10k rows, any endpoint or filter | 10.7 ms (budget 100 ms) |
| Live server, 8 clients, 1 worker | 323 req/s, p95 44 ms, 0 errors |
| Live server, 8 clients, 4 workers | 977 req/s, p95 18 ms, 0 errors |
| Ingest, concurrency 8, 0.5 s LLM latency | 837 items/min, 1.05 calls/item |
| Ingest 10k rows, fake LLM (no latency) | 10.7 s |
| Security | pip-audit 0 CVEs, bandit 0 findings, gitleaks 0 leaks |
