# Spec: Dynamic Competitive Pricing & Repricing Engine

**Status:** v1.1 — 2026-11-11 (implemented on branch `task/repricer-v1`; §11 lists decisions taken)
**Domain:** FinTech / E-Commerce Operations / Rule Systems
**Package:** `repricer/` (Python 3.11, FastAPI, Pydantic v2, SQLite via stdlib `sqlite3`, Anthropic SDK)
**Sibling spec:** `catalog-onboarding-taxonomy.spec.md` (same stack, same conventions)

---

## 1. Mission

Build an end-to-end repricing pipeline that:

1. Normalizes messy, unstructured competitor listing text into structured offers.
2. Computes bounded repricing recommendations from a deterministic policy rules engine.
3. Enforces safety margins and guardrails in code — never in the LLM.
4. Records every recommendation, approval, rejection and applied price change in an
   append-only audit trail.
5. Exposes a review endpoint so a human can inspect, approve or reject proposed changes
   before they take effect.

### 1.1 Non-negotiable design principle

**The LLM is an extraction tool, not a calculator.**

- The LLM sees raw scrape text and returns *entities* (price, coupon, condition, currency,
  SKU hint, confidence). Nothing else.
- All arithmetic — effective competitor price, margin floor, margin check, target price,
  clamping, rounding — runs in plain Python using `decimal.Decimal`. No `float` on the
  money path.
- LLM output is validated by a strict Pydantic schema and then re-validated against
  the source text (see §4.3). An extraction that fails validation is stored as
  `status='rejected'` with a reason; it never reaches the rules engine.
- If the Anthropic API is unavailable the pipeline degrades to a regex-only extractor
  with lower confidence; it never blocks or silently skips SKUs.

---

## 2. Scope

### In scope
- SQLite schema, migrations, seed data for `inventory` and `competitor_scrapes`.
- Extraction stage (LLM + regex fallback) producing `competitor_offers`.
- Rules engine with YAML-declared policies and deterministic evaluation.
- Margin safety checks and hard guardrails.
- Recommendation + review + apply flow with full audit log.
- FastAPI service: review endpoints, run trigger, audit query.
- CLI (`python -m repricer`) for `migrate`, `seed`, `extract`, `recommend`, `apply`, `run`.
- Tests: unit, property-based (Hypothesis) on the math, API tests, golden extraction fixtures.

### Out of scope (v1)
- Live scraping of competitor sites (scrapes arrive pre-loaded in the table).
- Multi-currency conversion (non-USD offers are extracted, flagged, and excluded from pricing).
- Auto-apply. Every price change goes through human review (decided 2026-11-11).
- Demand elasticity / ML-driven pricing. Rules only.
- Auth on the API. Single-tenant, internal tool. Add a header-token guard behind a flag only.

---

## 3. Data Model

All money columns are stored as `INTEGER` **cents** (USD). Percentages are stored as
`INTEGER` basis points (bp; 1500 bp = 15.00%). This removes floating-point from storage
entirely. Pydantic models expose `Decimal` on the API boundary and convert at the edge.

Every table has `created_at TEXT NOT NULL` (ISO-8601 UTC). Mutable tables add `updated_at`.
Audit tables are **append-only**: no `UPDATE`/`DELETE` statements exist against them in code,
and a SQLite trigger raises on any attempt.

### 3.1 `inventory` (given, normalized)

| column               | type    | notes                                          |
|----------------------|---------|------------------------------------------------|
| sku                  | TEXT PK |                                                |
| title                | TEXT    |                                                |
| cost_price_cents     | INTEGER | > 0                                            |
| min_margin_bp        | INTEGER | ≥ 0; floor margin over cost                    |
| current_price_cents  | INTEGER | > 0; the live price                            |
| stock_count          | INTEGER | ≥ 0                                            |
| price_class          | TEXT    | FK → `pricing_policies.price_class`; default `'default'` |
| updated_at           | TEXT    |                                                |

The source table arrives with `cost_price`, `min_margin_percent`, `current_price` as REAL.
Migration `0001` converts them to the integer columns above with `ROUND(x * 100)` after
asserting every REAL is a 2-decimal value (up to float noise); a sub-cent value such as
`12.344` aborts the migration rather than being rounded. The original REAL columns are dropped.

### 3.2 `competitor_scrapes` (given)

| column       | type       | notes                                             |
|--------------|------------|---------------------------------------------------|
| scrape_id    | INTEGER PK |                                                   |
| sku          | TEXT       | nullable — may need to be inferred from text      |
| competitor   | TEXT       | source name                                       |
| raw_text     | TEXT       | e.g. `"Item X on sale for $42.99 with $5 coupon"` |
| scraped_at   | TEXT       |                                                   |
| extracted    | INTEGER    | 0/1, set after extraction attempt (success or fail)|

### 3.3 `competitor_offers` (extraction output, one row per scrape attempt)

| column                 | type       | notes                                              |
|------------------------|------------|----------------------------------------------------|
| offer_id               | INTEGER PK |                                                    |
| scrape_id              | INTEGER FK | → `competitor_scrapes`                             |
| sku                    | TEXT FK    | → `inventory`; resolved SKU                        |
| competitor             | TEXT       |                                                    |
| listed_price_cents     | INTEGER    | sticker price as written                           |
| coupon_cents           | INTEGER    | absolute coupon, default 0                         |
| coupon_bp              | INTEGER    | percentage coupon, default 0                       |
| shipping_cents         | INTEGER    | nullable; `NULL` = unknown, `0` = free             |
| effective_price_cents  | INTEGER    | **computed in code**, see §4.4                     |
| currency               | TEXT       | ISO-4217; non-`USD` excluded from pricing          |
| condition              | TEXT       | `new` / `used` / `refurbished` / `unknown`         |
| in_stock               | INTEGER    | nullable 0/1                                       |
| extractor              | TEXT       | `llm:<model-id>` or `regex:v1`                     |
| confidence             | INTEGER    | 0–100                                              |
| status                 | TEXT       | `accepted` / `rejected` / `needs_review`           |
| reject_reason          | TEXT       | nullable                                           |
| raw_llm_json           | TEXT       | verbatim model output, for audit                   |
| created_at             | TEXT       |                                                    |

Unique: `(scrape_id)` — one offer per scrape. Re-running extraction on an already-extracted
scrape is a no-op unless `--force`, which inserts a new row and marks the old one
`status='superseded'` (the one permitted update on this table).

### 3.4 `pricing_policies` (loaded from `repricer/rules/policies.yaml`)

| column                   | type    | notes                                                   |
|--------------------------|---------|---------------------------------------------------------|
| price_class              | TEXT PK | e.g. `default`, `clearance`, `premium`                  |
| strategy                 | TEXT    | `match_lowest` / `undercut_lowest` / `hold` / `target_margin` |
| undercut_cents           | INTEGER | used by `undercut_lowest`                               |
| undercut_bp              | INTEGER | alternative to cents; engine uses `max(cents, bp-derived)` |
| max_decrease_bp          | INTEGER | per-run cap on price drop vs `current_price`            |
| max_increase_bp          | INTEGER | per-run cap on price rise vs `current_price`            |
| min_competitor_confidence| INTEGER | offers below this are ignored                           |
| max_offer_age_hours      | INTEGER | stale offers ignored                                    |
| low_stock_threshold      | INTEGER | stock ≤ threshold → suppress decreases                  |
| round_to_cents           | INTEGER | e.g. `99` → price ends in `.99`; `0` = no rounding      |
| version                  | INTEGER | bumped on every YAML reload; recorded on recommendations|

### 3.5 `repricing_runs`

| column          | type       | notes                                         |
|-----------------|------------|-----------------------------------------------|
| run_id          | TEXT PK    | ULID                                          |
| started_at      | TEXT       |                                               |
| finished_at     | TEXT       | nullable while running                        |
| trigger         | TEXT       | `cli` / `api` / `schedule`                    |
| policy_version  | INTEGER    |                                               |
| skus_considered | INTEGER    |                                               |
| recommendations | INTEGER    |                                               |
| status          | TEXT       | `running` / `completed` / `failed`            |
| error           | TEXT       | nullable                                      |

### 3.6 `price_recommendations` (append-only)

| column                   | type       | notes                                                   |
|--------------------------|------------|---------------------------------------------------------|
| recommendation_id        | TEXT PK    | ULID                                                    |
| run_id                   | TEXT FK    |                                                         |
| sku                      | TEXT FK    |                                                         |
| current_price_cents      | INTEGER    | snapshot at time of recommendation                      |
| cost_price_cents         | INTEGER    | snapshot                                                |
| min_margin_bp            | INTEGER    | snapshot                                                |
| floor_price_cents        | INTEGER    | computed: `ceil(cost × (1 + min_margin))`               |
| lowest_competitor_cents  | INTEGER    | nullable if no eligible offers                          |
| competitor_offer_ids     | TEXT       | JSON array of `offer_id`s that were eligible            |
| raw_target_cents         | INTEGER    | strategy output before bounds                           |
| proposed_price_cents     | INTEGER    | after bounds, floor, rounding                           |
| delta_cents              | INTEGER    | `proposed − current`                                    |
| delta_bp                 | INTEGER    |                                                         |
| decision                 | TEXT       | `propose` / `hold` / `blocked`                          |
| reason_codes             | TEXT       | JSON array, ordered, e.g. `["UNDERCUT","CLAMP_MAX_DECREASE","FLOOR_APPLIED"]` |
| explanation              | TEXT       | human-readable, generated from reason codes (templated, not LLM) |
| policy_version           | INTEGER    |                                                         |
| review_status            | TEXT       | `pending` / `approved` / `rejected` / `applied` / `expired` |
| created_at               | TEXT       |                                                         |

`review_status` is the single mutable column, and only via the state machine in §6.3.
All other columns are immutable (trigger-enforced).

### 3.7 `review_actions` (append-only)

| column              | type       | notes                                   |
|---------------------|------------|-----------------------------------------|
| action_id           | TEXT PK    | ULID                                    |
| recommendation_id   | TEXT FK    |                                         |
| action              | TEXT       | `approve` / `reject` / `override`       |
| actor               | TEXT       | free text / header value                |
| override_price_cents| INTEGER    | nullable; only for `override`           |
| note                | TEXT       | nullable                                |
| created_at          | TEXT       |                                         |

An `override` is re-validated against floor and bounds like any proposal. A reviewer
cannot push a price below the margin floor through the API; that path returns 422.

### 3.8 `price_changes` — the audit log (append-only)

| column              | type       | notes                                                |
|---------------------|------------|------------------------------------------------------|
| change_id           | TEXT PK    | ULID                                                 |
| sku                 | TEXT FK    |                                                      |
| recommendation_id   | TEXT FK    | nullable for manual/migration changes                |
| action_id           | TEXT FK    | nullable                                             |
| old_price_cents     | INTEGER    |                                                      |
| new_price_cents     | INTEGER    |                                                      |
| source              | TEXT       | `review_approve` / `auto_apply` / `override` / `seed` |
| actor               | TEXT       |                                                      |
| applied_at          | TEXT       |                                                      |
| prev_change_id      | TEXT       | nullable; hash-chain pointer                         |
| row_hash            | TEXT       | SHA-256 over canonical JSON of this row + `prev_change_id` |

`inventory.current_price_cents` is only ever written inside the same transaction that
inserts a `price_changes` row. There is one function (`apply_change()`) that does both;
nothing else touches that column after seeding. `repricer audit verify` walks the chain
per SKU and fails on any break.

---

## 4. Stage 1 — Unstructured Data Extraction

### 4.1 Input
Rows from `competitor_scrapes` where `extracted = 0`. Batched (default 20 per LLM call).

### 4.2 LLM extractor
- Model: latest Claude (configured in `repricer/config.py`, default `claude-sonnet-5-5`;
  overridable via `REPRICER_MODEL`).
- Uses structured outputs (`output_config.format`, JSON schema) — not forced tool use, which
  current models reject. No free text. Amounts are decimal strings.
- System prompt forbids arithmetic: *"Report prices exactly as written. Do not subtract
  coupons, do not compute totals, do not convert currencies."*
- Input to the model is **only** `raw_text` and the `scrape_id`. The model is never shown
  `cost_price`, `min_margin`, or `current_price`.
- No sampling parameters (rejected by current models); determinism for tests comes from
  recorded cassettes. Prompt-cached system prompt. SDK retries ×3 on 429/5xx, then fallback.

`ExtractedOffer` schema (Pydantic, `extra='forbid'`):

```
scrape_id: int
sku_hint: str | None          # SKU or product identifier visible in text
listed_price: Decimal         # as written, ≥ 0, ≤ 2 dp
currency: Literal["USD","CAD","EUR","GBP","OTHER","UNKNOWN"]
coupon_amount: Decimal | None # absolute, as written
coupon_percent: Decimal | None
shipping: Decimal | None      # None = not mentioned; 0 = explicitly free
condition: Literal["new","used","refurbished","unknown"]
in_stock: bool | None
confidence: int               # 0–100, model's self-report; capped at 90 in code
```

### 4.3 Post-extraction validation (code, not LLM)
A `ValidationResult` is computed per offer. Any failure → `status='rejected'` with
`reject_reason`:

| check                                          | reason code             |
|------------------------------------------------|-------------------------|
| `listed_price` not found as a substring of `raw_text` (after normalizing `$`, `,`, `USD`) | `PRICE_NOT_IN_SOURCE` |
| `coupon_amount` not found in source            | `COUPON_NOT_IN_SOURCE`  |
| `listed_price == 0`                            | `ZERO_PRICE`            |
| `listed_price > 100 × inventory.current_price` | `IMPLAUSIBLE_PRICE`     |
| SKU unresolvable (no `scrapes.sku` and no `sku_hint` match in inventory) | `SKU_UNRESOLVED` |
| `currency != USD`                              | `NON_USD` (stored, excluded from pricing, not rejected) |
| both `coupon_amount` and `coupon_percent` set  | `AMBIGUOUS_COUPON` → `needs_review` |

SKU resolution order: `scrapes.sku` → exact `sku_hint` match → case-insensitive title
match with ≥ 0.9 token-set ratio (stdlib `difflib`; no fuzzy lib dep). Anything weaker is
`SKU_UNRESOLVED`.

### 4.4 Effective price (deterministic)

```
effective = listed_price
          − coupon_amount                       (if set)
          − round_half_even(listed_price × coupon_percent / 100)   (if set)
          + shipping                            (if shipping is not None)
effective = max(effective, 0)
```

All in `Decimal`, quantized to cents with `ROUND_HALF_EVEN`. Shipping `None` is treated as
`0` **and** adds reason code `SHIPPING_UNKNOWN` to any recommendation relying on it.

### 4.5 Regex fallback (`regex:v1`)
Patterns for `$X.XX`, `X.XX USD`, `$X off`, `X% off`, `free shipping`, `+ $X shipping`.
Confidence fixed at 40. Used when the API is unreachable after retries or
`REPRICER_EXTRACTOR=regex`. Same validation pipeline as §4.3.

### 4.6 Golden fixtures
`tests/fixtures/scrapes/*.json` — ≥ 40 raw snippets with expected `ExtractedOffer`.
Cover: plain price, coupon $, coupon %, both (ambiguous), free shipping, paid shipping,
"was/now" strike-through pricing (must extract *now* price), bundle pricing ("2 for $50" →
`needs_review`), non-USD, price-per-unit, typo'd currency symbols, no price at all.

The LLM test suite runs against recorded responses (`tests/fixtures/llm_cassettes/`).
`REPRICER_LLM_MODE=live|record|replay` selects real calls, real calls + save, or replay only.

---

## 5. Stage 2 — Pricing Policy Rules Engine

### 5.1 Inputs per SKU
- `inventory` row (snapshot).
- Eligible offers: `status='accepted'`, `currency='USD'`, `condition='new'`,
  `confidence ≥ policy.min_competitor_confidence`,
  `scraped_at ≥ now − policy.max_offer_age_hours`, `in_stock != 0`.
- Policy for `inventory.price_class`.

### 5.2 Evaluation order (fixed, documented, tested)

```
1. floor         = ceil_cents(cost × (1 + min_margin_bp / 10000))
2. lowest        = min(effective_price over eligible offers)   → None if no offers
3. raw_target    = strategy(lowest, current, policy)            → see 5.3
4. bounded       = clamp(raw_target,
                         lo = current × (1 − max_decrease_bp/10000),
                         hi = current × (1 + max_increase_bp/10000))
5. floored       = max(bounded, floor)
6. stock guard   = if stock ≤ low_stock_threshold and floored < current → floored = current
7. rounded       = round_to(floored, policy.round_to_cents)     → only if floored != current;
                                                                  never below floor or the step-4
                                                                  lower bound; rounds up instead
8. decision      = propose if rounded != current
                   hold    if rounded == current
                   blocked if any hard guardrail fired (5.4)
```

Every step that alters the value appends a reason code. The final `reason_codes` list is
the complete derivation. `explanation` is rendered from a template per code — no LLM.

### 5.3 Strategies

| strategy        | raw_target                                                          |
|-----------------|---------------------------------------------------------------------|
| `match_lowest`  | `lowest` ; if no offers → `current` (`NO_COMPETITOR_DATA`)          |
| `undercut_lowest` | `lowest − max(undercut_cents, lowest × undercut_bp/10000)`        |
| `hold`          | `current` always                                                    |
| `target_margin` | `ceil_cents(cost × (1 + target_margin_bp/10000))`; ignores competitors |

### 5.4 Hard guardrails (→ `decision='blocked'`, never proposed, never auto-applied)

| code                      | condition                                                        |
|---------------------------|------------------------------------------------------------------|
| `FLOOR_ABOVE_CURRENT`     | `floor > current` — the SKU is *already* under margin; needs human |
| `COST_ANOMALY`            | `cost_price_cents <= 0` or `cost > current × 10`                 |
| `SINGLE_SOURCE_LOW_CONF`  | only one eligible offer and its confidence < 70                  |
| `DELTA_EXCEEDS_HARD_CAP`  | `|delta_bp| > HARD_CAP_BP` (global, default 5000, not policy-overridable) |
| `PRICE_OUT_OF_RANGE`      | proposed < $0.01 or > $1,000,000                                 |

### 5.5 Policy YAML
`repricer/rules/policies.yaml`, validated by a Pydantic model on load. Reload bumps
`version`. Example:

```yaml
default:
  strategy: undercut_lowest
  undercut_cents: 50
  undercut_bp: 100
  max_decrease_bp: 1000
  max_increase_bp: 500
  min_competitor_confidence: 60
  max_offer_age_hours: 72
  low_stock_threshold: 3
  round_to_cents: 99
clearance:
  strategy: match_lowest
  max_decrease_bp: 3000
  max_increase_bp: 0
  ...
```

### 5.6 Invariants (Hypothesis property tests)
For all random `cost > 0`, `min_margin_bp ≥ 0`, `current > 0`, any offer set, any valid policy:

- `proposed ≥ floor` **always**.
- `proposed ≥ cost` **always** (corollary).
- `|proposed − current| / current ≤ max(max_decrease_bp, max_increase_bp)/10000 + rounding slack`
  unless `FLOOR_APPLIED` lifted it.
- `decision == 'blocked'` ⇒ `proposed_price_cents == current_price_cents`.
- Output is a pure function of `(inventory_row, offers, policy, now)`; running twice yields
  identical rows except `recommendation_id` and `created_at`.
- No `float` appears anywhere in `repricer/engine/` (enforced by a test that greps the AST
  for `float` calls and `/` on non-Decimal operands).

---

## 6. Stage 3 — Execution & Audit Trail

### 6.1 Run
`repricer run` = `extract` → `recommend` in one `repricing_runs` record. Each stage is
idempotent; a crash mid-run leaves `status='failed'` with `error` and no partial
`price_changes`.

### 6.2 Apply
`apply_change(recommendation_id, action_id, actor, source)`:

```
BEGIN IMMEDIATE
  rec = SELECT ... FROM price_recommendations WHERE id=? AND review_status IN ('approved')
  inv = SELECT ... FROM inventory WHERE sku=? 
  assert inv.current_price_cents == rec.current_price_cents   -- else STALE_SNAPSHOT, abort
  re-run guardrails against *live* inventory row (cost/margin may have changed)
  INSERT price_changes (..., prev_change_id = last change for sku, row_hash = sha256(...))
  UPDATE inventory SET current_price_cents=?, updated_at=? WHERE sku=?
  UPDATE price_recommendations SET review_status='applied' WHERE id=?
COMMIT
```

`STALE_SNAPSHOT` → recommendation moves to `expired`; a new run is required.

### 6.3 Review state machine

```
pending ──approve──▶ approved ──apply──▶ applied
   │                    │
   ├──reject──▶ rejected│
   ├──override─▶ approved (with override_price, re-validated)
   └──(new run for same sku)──▶ expired
```

Illegal transitions return 409 from the API and raise `IllegalTransition` from the service.

### 6.4 Auto-apply

Removed from v1. Every change is reviewed by a human.

### 6.5 Audit queries
- `repricer audit sku <SKU>` — full price history with recommendation + action chain.
- `repricer audit verify` — recompute every `row_hash`, follow `prev_change_id`, exit 1 on break.
- `repricer audit run <run_id>` — all recommendations with reason codes.

---

## 7. Review Endpoint (FastAPI)

Base path `/v1`. JSON only. Money as `Decimal` strings (`"42.99"`), never floats.

| method | path                                   | purpose                                               |
|--------|----------------------------------------|-------------------------------------------------------|
| POST   | `/runs`                                | trigger a run (`{"stages": ["extract","recommend"]}`); 202 + `run_id` |
| GET    | `/runs/{run_id}`                       | run status + counts                                   |
| GET    | `/recommendations`                     | list; filters `status`, `sku`, `run_id`, `decision`, `min_abs_delta_bp`; paginated |
| GET    | `/recommendations/{id}`                | full record incl. eligible offers, reason codes, explanation |
| POST   | `/recommendations/{id}/approve`        | body `{"actor": "...", "note": "..."}` → 200; applies immediately |
| POST   | `/recommendations/{id}/reject`         | body `{"actor", "note"}`                               |
| POST   | `/recommendations/{id}/override`       | body `{"actor", "price": "41.99", "note"}`; 422 if below floor or outside hard cap |
| GET    | `/skus/{sku}/history`                  | `price_changes` chain for SKU                         |
| GET    | `/skus/{sku}/offers`                   | current eligible + rejected offers with reasons        |
| GET    | `/audit/verify`                        | `{ "ok": bool, "breaks": [...] }`                      |
| GET    | `/policies`                            | active policies + version                              |
| POST   | `/policies/reload`                     | re-read YAML; 422 on schema error, old version stays   |
| GET    | `/healthz`                             |                                                        |

Error envelope: `{"error": {"code": "FLOOR_VIOLATION", "message": "...", "details": {...}}}`.
Status codes: 404 unknown id, 409 illegal transition / stale snapshot, 422 validation.

The `GET /recommendations/{id}` response is the "review card": it must contain enough for
a human to decide without opening the DB — current, proposed, floor, lowest competitor,
every eligible offer's raw text + effective price, and the ordered reason codes with
explanations.

---

## 8. Package Layout

```
repricer/
  __init__.py
  __main__.py            # CLI: migrate | seed | extract | recommend | apply | run | audit | serve
  config.py              # env-driven settings (DB path, model, extractor mode, hard caps)
  money.py               # Cents / Bp newtypes, Decimal helpers, round_to, ceil_cents
  models.py              # Pydantic: ExtractedOffer, Recommendation, ReviewCard, ...
  db/
    schema.sql           # all tables + append-only triggers
    migrations/0001_normalize_money.sql
    repo.py              # thin typed query layer; the only module that writes SQL
  extraction/
    llm.py               # Anthropic client, tool schema, batching, retries, cassettes
    regex_extractor.py
    validate.py          # §4.3 checks + SKU resolution
    effective_price.py   # §4.4
  engine/
    policy.py            # YAML load + Pydantic validation + version
    strategies.py
    guardrails.py
    evaluate.py          # §5.2 pipeline; pure function
    explain.py           # reason-code → text templates
  execution/
    runner.py            # repricing_runs lifecycle
    apply.py             # apply_change(), hash chain
    review.py            # state machine
  api/
    app.py
    routes_recommendations.py
    routes_runs.py
    routes_audit.py
  rules/
    policies.yaml
tests/
  unit/ (money, effective_price, strategies, guardrails, evaluate, review_sm, hash_chain)
  property/ (hypothesis invariants §5.6)
  extraction/ (golden fixtures, validation, regex fallback, cassette replay)
  api/ (httpx TestClient, full approve/reject/override flows, error envelopes)
  fixtures/
```

---

## 9. Acceptance Criteria

Each AC is a test or a command that must pass.

**Extraction**
- AC-E1: All ≥ 40 golden fixtures extract to the expected `ExtractedOffer` via cassette replay.
- AC-E2: A fabricated price not present in `raw_text` is rejected with `PRICE_NOT_IN_SOURCE`.
- AC-E3: With `REPRICER_EXTRACTOR=regex`, the pipeline completes with no network and every
  offer has `extractor='regex:v1'`, `confidence=40`.
- AC-E4: The LLM request payload never contains the strings `cost`, `margin`, or any
  `inventory` price (asserted on the recorded request).
- AC-E5: Re-running `extract` on already-extracted scrapes inserts zero new offer rows.

**Rules engine**
- AC-R1: Hypothesis invariants in §5.6 pass with ≥ 2,000 examples each.
- AC-R2: Given cost $30.00, min margin 20%, current $45.00, competitor $42.99 − $5 coupon,
  policy `undercut_lowest` (50¢, max_decrease 10%, round .99): floor = $36.00,
  lowest = $37.99, raw = $37.49, clamped to $40.50, rounded to $40.99, decision `propose`,
  reason codes `["UNDERCUT","CLAMP_MAX_DECREASE","ROUNDED"]` (plus `SHIPPING_UNKNOWN` after
  `UNDERCUT` when the scrape text does not mention shipping, as the brief's example does not).
- AC-R3: Same as R2 but current $35.00 → `blocked`, `FLOOR_ABOVE_CURRENT`.
- AC-R4: Stock = 2 with threshold 3 → `hold`, `LOW_STOCK_NO_DECREASE`.
- AC-R5: No `float` in `repricer/engine/` or `repricer/money.py` (AST test).
- AC-R6: `evaluate()` is deterministic: two calls with identical input produce identical
  output (excluding ids/timestamps).

**Execution & audit**
- AC-X1: Approving a recommendation produces exactly one `price_changes` row and updates
  `inventory.current_price_cents` in the same transaction; forcing an exception between
  the two leaves both unchanged.
- AC-X2: `repricer audit verify` passes on a fresh DB after 100 random approvals; tampering
  with one `new_price_cents` via raw SQL makes it exit 1 and name the SKU.
- AC-X3: Any `UPDATE`/`DELETE` on `price_changes` or `review_actions` raises from the
  SQLite trigger.
- AC-X4: Approving a recommendation whose `current_price_cents` snapshot no longer matches
  inventory returns 409 and marks it `expired`.
- AC-X5: A `hold` or `blocked` recommendation cannot be approved (409).

**Review API**
- AC-A1: Full flow test: run → list pending → approve one → history shows change → verify OK.
- AC-A2: Override below floor → 422 `FLOOR_VIOLATION`; override within bounds → applied with
  `source='override'`.
- AC-A3: All money fields in responses are strings with exactly 2 decimals; no response
  body contains a JSON number for a price.
- AC-A4: `POST /policies/reload` with invalid YAML → 422 and `GET /policies` still shows the
  previous version.

**Hygiene**
- `ruff check`, `mypy --strict repricer`, `pytest -q` all green.
- `python -m repricer migrate && seed && run` works on an empty DB in < 60 s with the
  regex extractor.

---

## 10. Assumptions & Open Questions

The brief left the four "Core Functional Requirements" headings empty. This spec fills
them with the above; confirm or redirect:

1. **Currency.** Assumed single-currency USD. Non-USD offers are stored but ignored. OK?
2. **Auto-apply.** Defaulted to off with a per-policy opt-in and a small delta cap.
   Should v1 ship with auto-apply removed entirely?
3. **Hard cap.** Global `HARD_CAP_BP=5000` (±50%) is not policy-overridable. Right number?
4. **Condition.** Only `new` competitor offers count. Should `refurbished` count at a discount?
5. **Price increases.** Engine can propose increases when competitors are higher
   (`max_increase_bp`). Is that in scope or should v1 be decrease-only?
6. **Seed data.** Is a `competitor_scrapes` dataset provided, or do we generate ~200
   synthetic snippets (deterministic seed) for development and the golden fixtures?
7. **Auth.** None assumed. If this is exposed beyond localhost, a static bearer token
   behind `REPRICER_API_TOKEN` is the minimum.

---

## 11. Decisions taken during implementation (2026-11-11)

| topic | decision |
|-------|----------|
| Auto-apply | Removed from v1 (CTO). |
| Price increases | Allowed, bounded by `max_increase_bp` (CTO). |
| Seed data | Deterministic generator, seed 42, 50 SKUs / 200 scrapes (CTO). |
| LLM tests | Cassette replay; `REPRICER_LLM_MODE` replaces `REPRICER_LIVE_LLM` (CTO). |
| Model | `claude-sonnet-5-5` default, structured outputs, no forced tool choice, no temperature. |
| ULID | stdlib implementation in `repricer/ids.py`; no extra dependency. |
| Default policy confidence | `min_competitor_confidence: 40` so regex offers (40) qualify; `premium` keeps 60. |
| Rounding | Skipped when the price does not move; floored at `max(floor, lower bound)`. |
| Override | Allowed on `pending` recs of any decision (can rescue `FLOOR_ABOVE_CURRENT`), re-validated. |
| Cassettes | Not recorded yet: no API key in the build environment. `scripts/record_cassettes.py` ready. |
