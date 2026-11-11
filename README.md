# repricer — Dynamic Competitive Pricing & Repricing Engine

Normalizes messy competitor scrape text into structured offers, computes bounded repricing
recommendations with a deterministic rules engine, enforces margin guardrails in code, and
keeps an append-only, hash-chained audit trail. Humans approve every price change through
a review API.

Spec: [`docs/specs/dynamic-repricing-engine.spec.md`](docs/specs/dynamic-repricing-engine.spec.md).

## Design in one paragraph

The LLM is an **extraction tool, not a calculator**. It sees only the raw scrape text and
returns entities (listed price, coupon, shipping, condition, currency) as decimal strings.
Code then checks that every extracted amount literally appears in the source, computes the
effective price in `Decimal`, and runs a fixed eight-step pricing pipeline on integer cents.
A test walks the AST of the engine and money modules and fails on any `float`. Guardrails
*block* rather than propose. The only code path that changes a live price also appends a
`price_changes` row with a SHA-256 chain pointer, in the same transaction.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'

export REPRICER_DB_PATH=dev.db
export REPRICER_EXTRACTOR=regex        # no API key needed; use llm for Claude

python -m repricer seed                # writes the brief's REAL-money schema, then migrates it
python -m repricer run                 # extract -> recommend, prints a few proposals
python -m repricer serve               # http://127.0.0.1:8000/docs
```

Review one through the API:

```bash
curl -s 'localhost:8000/v1/recommendations?status=pending&decision=propose&limit=3'
curl -s localhost:8000/v1/recommendations/<id>              # the review card
curl -s -X POST localhost:8000/v1/recommendations/<id>/approve \
  -H 'content-type: application/json' -d '{"actor":"alex","note":"ok"}'
curl -s localhost:8000/v1/skus/SKU-0001/history
python -m repricer audit verify
```

### With Claude

```bash
export ANTHROPIC_API_KEY=...
export REPRICER_EXTRACTOR=llm
export REPRICER_MODEL=claude-sonnet-5-5     # default
python -m repricer extract --force          # re-extract all scrapes with the LLM
python -m repricer recommend
```

If the API is unavailable (5xx, rate limited, refusal), the batch falls back to the regex
extractor at confidence 40 and the run continues. Nothing is skipped silently.

### Record LLM cassettes for tests

Tests replay recorded responses so CI needs no key. To record (once, or when the prompt or
fixtures change):

```bash
REPRICER_LLM_MODE=record ANTHROPIC_API_KEY=... python scripts/record_cassettes.py
```

Commit `tests/fixtures/llm_cassettes/*.json`. Until recorded, `test_ac_e1_golden_llm_replay`
skips with an explicit message.

## Layout

```
repricer/
  money.py            integer cents / basis points, Decimal at the edges, no floats
  models.py           domain models, reason codes, enums
  db/                 schema.sql (append-only triggers), migrate.py, repo.py (all SQL)
  extraction/         llm.py, regex_extractor.py, validate.py, effective_price.py, pipeline.py
  engine/             evaluate.py (8 fixed steps), strategies.py, guardrails.py, explain.py
  execution/          review.py (state machine), apply.py (only price writer), audit.py
  api/                FastAPI app, routes, string-money schemas, error envelope
  rules/policies.yaml pricing policies by price_class
tests/                unit, property (Hypothesis), extraction (200 golden fixtures), api
```

## Evaluation order (engine)

1. floor = ceil(cost × (1 + min_margin))
2. lowest eligible competitor effective price
3. strategy → raw target
4. clamp to [current × (1 − max_decrease), current × (1 + max_increase)]
5. max(·, floor)
6. low stock ⇒ no decrease
7. round to policy ending, never below floor or the lower bound, only if price moves
8. decision: propose / hold / blocked (hard guardrails)

Every step that changes the value appends a reason code; the audit trail is the full
derivation.

## Checks

```bash
ruff check . && mypy && pytest -q
```

## Notes

- Money is stored as INTEGER cents and percentages as basis points. API responses give
  money as 2-decimal strings, never JSON numbers.
- SQLite `BEGIN IMMEDIATE` serializes applies. Fine for an internal review tool.
- No auth in v1; bind to localhost or put it behind a proxy.
- Title-based SKU resolution refuses on ambiguity (`SKU_UNRESOLVED`) rather than guessing.
