# catalog-onboard

Ingestion engine for third-party seller listings: cleans raw titles, maps items to the
canonical marketplace taxonomy, extracts required attributes, enforces compliance rules
deterministically, and routes each item to a **draft catalog** or **quarantine**. A read-only
FastAPI service exposes every outcome together with its full stage-by-stage trail.

Spec: [docs/specs/catalog-onboarding-taxonomy.spec.md](docs/specs/catalog-onboarding-taxonomy.spec.md)

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

# 1. synthetic starter DB (canonical_categories + incoming_supplier_feeds)
.venv/bin/python scripts/make_starter_db.py --out data/starter.sqlite --seed 7

# 2. ingest (fake LLM; set ANTHROPIC_API_KEY or --llm anthropic for the real model)
.venv/bin/python -m catalog_onboard ingest --db data/starter.sqlite --llm fake -v
.venv/bin/python -m catalog_onboard ingest --db data/starter.sqlite --llm fake   # no-op: already processed

# 3. inspect
.venv/bin/python scripts/serve.py --db data/starter.sqlite
#   GET /stats  /catalog/draft?q=shorts  /quarantine?reason_code=PROHIBITED_GLOBAL
#   GET /items/SAMPLE-0001/trace  /taxonomy/kitchen.cutlery.chef_knives  /docs

# tests, lint, types
.venv/bin/pytest -q          # add -m slow for the 10k-row latency check
.venv/bin/ruff check . && .venv/bin/mypy --strict catalog_onboard
```

## Pipeline

```
incoming_supplier_feeds
  -> clean       regex/dictionary normalisation -> NormalizedListing          (Python)
  -> candidates  BM25 shortlist of leaf categories, top-K                     (Python)
  -> classify    pick one id from the shortlist; fill *missing* attributes     (LLM, schema-constrained)
  -> reconcile   deterministic values win; defaults; conflicts -> warnings    (Python)
  -> validate    enum / range / length / pattern / multi checks               (Python)
  -> comply      prohibited keywords (inherited), price, title, duplicates    (Python)
  -> route       zero blocking codes -> draft_catalog, else quarantine        (Python)
```

Every stage writes to `pipeline_runs` (inputs, outputs, raw LLM responses, duration), so an
operator can see why an item landed where it did without re-running the model.

### The LLM boundary

| Concern | Owner |
|---|---|
| Cleaning, tag mapping, size/color/pack/price parsing | Python |
| Category shortlist | Python (BM25) |
| Pick category among shortlist | LLM, `Literal`-constrained output |
| Fill attributes the parser could not find | LLM, JSON schema generated from the category spec |
| Attribute validation, prohibited keywords, pricing, duplicates, routing | Python |
| "Is this listing compliant?" | never asked of the LLM |

The model's answer is parsed into a Pydantic model built per request from the candidate set and
the category's `required_attributes_json`. An out-of-vocabulary answer fails to parse, is retried
with the validator message (`LLM_MAX_RETRIES`, default 2), then quarantined as
`LLM_SCHEMA_INVALID` with the raw responses stored. Prompts never contain compliance vocabulary;
a test greps for it.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` if `ANTHROPIC_API_KEY` is set, else `fake` | `anthropic` or `fake` |
| `LLM_MODEL` | `claude-opus-5-5` | model id for the real client |
| `LLM_MAX_RETRIES` | `2` | re-prompts after a schema violation |
| `HIGH_CONF_THRESHOLD` | `0.75` | BM25 margin above which a disagreeing low-confidence LLM pick is quarantined |
| `DISAGREE_QUARANTINE_BELOW` | `0.8` | LLM confidence below which that disagreement blocks |
| `CANDIDATE_K` | `8` | shortlist size |
| `RULES_DIR` | `catalog_onboard/rules` | YAML rule tables, hashed into `rules_version` |

The real client uses the Anthropic SDK with structured JSON-schema output and server-side
refusal fallbacks enabled.

## Re-validation (AC-2.4)

```bash
# after editing canonical_categories.required_attributes_json / prohibited_keywords:
.venv/bin/python -m catalog_onboard revalidate --db data/starter.sqlite   # 0 LLM calls
```

Outcomes are versioned by rule files + taxonomy content, so a spec change supersedes old outcomes
(the trail in `pipeline_runs` is kept) and a re-run at the same version is a no-op.

## Reason codes

`blocking` codes quarantine; `warning`/`info` are kept on the draft row for review.

- clean: `TITLE_EMPTY`*, `PRICE_UNPARSEABLE`, `TAG_UNMAPPED`, `ATTR_CONFLICT:<key>`,
  `TEXT_ENCODING_INVALID:<field>` (field unset), `LANG_UNSUPPORTED` (letters outside the Latin script)
- classify: `CLASSIFICATION_NO_CANDIDATES`*, `CLASSIFICATION_WEAK_CANDIDATES`, `CLASSIFICATION_DISAGREE`(*),
  `CLASSIFICATION_NOT_LEAF`*, `CLASSIFICATION_OUT_OF_SET`*, `LLM_SCHEMA_INVALID`* and `LLM_TRANSPORT_ERROR`*
  (`detail` = `classify` or `extract`; the API filter `?reason_code=LLM_SCHEMA_INVALID` also matches the
  pre-v1.1 form `LLM_SCHEMA_INVALID:<stage>`)
- reconcile: `ATTR_DEFAULTED:<key>`, `ATTR_EXTRA:<key>`, `ATTR_CONFLICT:<key>`
- validate*: `ATTR_MISSING`, `ATTR_ENUM_INVALID`, `ATTR_RANGE`, `ATTR_TOO_LONG`, `ATTR_PATTERN`, `ATTR_MULTI_NOT_ALLOWED`
- comply*: `PROHIBITED_KEYWORD:<kw>`, `PROHIBITED_GLOBAL:<kw>`, `PRICE_INVALID`, `TITLE_TOO_SHORT`, `DUPLICATE_LISTING`
- engine: `PIPELINE_ERROR:<stage>`* (one bad row never aborts the batch)

## Idempotency

`(feed_item_id, rules_version)` is recorded in `processed_items`. Re-running with the same rules
is a no-op; changing any YAML rule file changes `rules_version`, supersedes the previous outcome,
and keeps the old trail in `pipeline_runs`.

## Layout

```
catalog_onboard/
  models.py config.py taxonomy.py cli.py
  rules/        abbreviations.yaml supplier_tag_map.yaml global_prohibited.yaml
  db/           schema.sql store.py
  pipeline/     clean.py extract.py candidates.py reconcile.py validate.py comply.py route.py engine.py
  pipeline/llm/ schema.py prompts.py runner.py fake.py anthropic_client.py
  api/          app.py routes.py pagination.py
scripts/        make_starter_db.py serve.py
tests/          one module per stage + API + CLI + end-to-end; fixtures/normalized_golden.json
```
