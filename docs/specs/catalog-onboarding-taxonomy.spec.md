# Spec: Automated Catalog Onboarding & Taxonomy Normalization

| | |
|---|---|
| Status | v1.1 (implementation decisions written back) |
| Date | 2026-11-12 (v1: 2026-11-10) |
| Dev plan | [`docs/plans/catalog-onboarding-taxonomy.dev-plan.md`](../plans/catalog-onboarding-taxonomy.dev-plan.md) |
| Domain | E-Commerce Catalog & Merchandising / Marketplace Operations |
| Branch | `taxonomy` |
| Package | `catalog_onboard/` (Python 3.11, Pydantic v2, FastAPI, SQLite) |

## 1. Mission

Build an ingestion engine that takes raw third-party seller listings from
`incoming_supplier_feeds`, normalizes titles and variations, maps each item to
the canonical marketplace taxonomy, extracts the attributes the target category
requires, enforces compliance rules deterministically, and publishes either to a
**draft catalog** or a **quarantine** queue. A read-only inspection API exposes
the result and the reasoning trail for every item.

## 2. Design principles (non-negotiable)

1. **LLM proposes, Python decides.** The LLM is used only for extraction and
   classification *candidates*. Every policy, compliance, and schema decision is
   made by deterministic Python code over typed data.
2. **Never trust a raw completion.** Every LLM response is parsed into a Pydantic
   model via structured output / JSON schema. A parse failure is a retry, then a
   quarantine reason, never a silent pass.
3. **Every outcome is explainable.** Each listing carries an ordered list of
   machine-readable reason codes that explain why it landed where it did.
4. **Idempotent and replayable.** Re-running the pipeline on the same feed row
   with the same ruleset version produces the same outcome and does not
   duplicate rows.
5. **Deterministic first, LLM second.** If a regex/dictionary can extract a
   value (size, color, pack count), use it. LLM fills the gaps and is
   cross-checked against the deterministic result.

## 3. Starter data (given)

### 3.1 `canonical_categories`

| column | type | notes |
|---|---|---|
| `category_id` | TEXT PK | stable id, e.g. `apparel.men.bottoms.shorts` |
| `name` | TEXT | display name |
| `parent_id` | TEXT NULL | FK → `category_id`; NULL = root |
| `required_attributes_json` | TEXT (JSON) | array of attribute specs, see 3.3 |
| `prohibited_keywords` | TEXT | pipe- or comma-delimited list; case-insensitive match on normalized title + bullets |

Leaf categories are the only valid classification targets. Non-leaf
categories are navigation nodes.

### 3.2 `incoming_supplier_feeds`

| column | type | notes |
|---|---|---|
| `feed_item_id` | TEXT PK | supplier-scoped unique id |
| `supplier_id` | TEXT | |
| `raw_title` | TEXT | e.g. `Men's Dry-Fit Athletic Gym Shorts w/ Pockets Pack of 2 (Black/Navy, L) - Running Workout` |
| `raw_bullets` | TEXT | newline / `•` / `;` separated, inconsistent |
| `raw_tags` | TEXT | supplier-specific `size=XL;colour=NVY` style tags |
| `raw_price` | TEXT | may include currency symbols, ranges |
| `received_at` | TEXT (ISO-8601) | |

Any other columns present in the starter DB are passed through untouched into
`raw_json` on the listing record.

### 3.3 `required_attributes_json` contract

```json
[
  {"key": "size",        "type": "enum",   "allowed": ["XS","S","M","L","XL","XXL"], "multi": true},
  {"key": "color",       "type": "enum",   "allowed": ["black","navy","white", "..."], "multi": true},
  {"key": "gender",      "type": "enum",   "allowed": ["men","women","unisex","kids"]},
  {"key": "pack_count",  "type": "int",    "min": 1, "max": 100},
  {"key": "material",    "type": "string", "max_len": 64}
]
```

Supported `type` values: `enum`, `int`, `float`, `string`, `bool`. An attribute
spec may carry `allowed`, `min`, `max`, `max_len`, `multi`, `pattern`.
Unknown spec fields are an **ingestion configuration error** (fail fast at
startup, not per item).

## 4. Pipeline

```
incoming_supplier_feeds
        │
        ▼
 [1] Clean & Normalize ──────────────► NormalizedListing (Pydantic)
        │
        ▼
 [2] Deterministic Extraction ───────► DeterministicAttrs
        │
        ▼
 [3] LLM Extraction + Classification ─► LLMProposal (schema-validated, ≤N retries)
        │
        ▼
 [4] Reconcile ──────────────────────► CandidateListing (attrs + category + confidence)
        │
        ▼
 [5] Deterministic Validation & Compliance ─► ValidationResult (reason codes)
        │
   ┌────┴────┐
   ▼         ▼
draft_catalog   quarantine
        │
        ▼
 [6] Inspection API (read-only)
```

Every stage writes its output to `pipeline_runs` so the trail can be
re-inspected without re-running the LLM.

## 5. Functional requirements and acceptance criteria

### FR-1 Unstructured data extraction & cleaning

**Scope:** Turn `raw_title`, `raw_bullets`, `raw_tags` into a
`NormalizedListing`.

Rules (deterministic, in this order):

1. Unicode NFKC normalize; collapse whitespace; strip control chars.
2. Expand a fixed abbreviation dictionary: `w/`→`with`, `pk`/`pack of`→
   pack count, `mens`/`men's`→`men`, `wmn`→`women`, etc. Dictionary lives in
   `rules/abbreviations.yaml`, versioned.
3. Split bullets on `\n`, `•`, `;`, ` - ` (when surrounded by spaces); drop
   empty and duplicate bullets (case-insensitive).
4. Parse `raw_tags` as `key=value` pairs with `;`/`,`/`|` separators; map
   supplier keys to canonical keys via `rules/supplier_tag_map.yaml`
   (per-supplier, e.g. `colour`→`color`, `NVY`→`navy`).
5. Extract parenthetical variation groups `(Black/Navy, L)` into candidate
   color/size lists.
6. Extract pack count from `Pack of N`, `N-Pack`, `Nx`, `Set of N`.
7. Price: strip currency, parse decimal; ranges yield `price_min`/`price_max`.
8. Produce `clean_title`: original with variation parentheticals, pack
   phrases, and trailing ` - keyword dump` segments removed, title-cased,
   ≤ 150 chars.

Acceptance criteria:

- AC-1.1 Given the sample title in 3.2, `clean_title` equals
  `Men Dry-Fit Athletic Gym Shorts With Pockets`, `pack_count == 2`,
  colors `{black, navy}`, sizes `{L}`, and `keyword_tail ==
  ["Running", "Workout"]`.
- AC-1.2 Bullets containing only whitespace or exact duplicates are removed;
  order of the remaining bullets is preserved.
- AC-1.3 Supplier tag `colour=NVY` for a supplier with a mapping yields
  `color: navy`; with no mapping the tag is retained under `unmapped_tags`
  and reason code `TAG_UNMAPPED` is recorded (warning, not blocking).
- AC-1.4 Cleaning is pure: same input + same rules version → byte-identical
  `NormalizedListing`. Covered by a snapshot test over ≥ 50 fixture rows.
- AC-1.5 Malformed input (empty title, non-UTF8 bytes, price not parseable)
  never raises; it yields a `NormalizedListing` with the field unset and a
  reason code (`TITLE_EMPTY`, `PRICE_UNPARSEABLE`,
  `TEXT_ENCODING_INVALID:<field>`). Undecodable bytes are read with
  replacement at the DB connection, so reading a row can never abort the
  batch; the affected field is then unset by the clean stage.
- AC-1.6 No LLM call occurs in this stage (asserted in tests by a mock
  client that fails on invocation).

### FR-2 Taxonomy classification & attribute validation

**Scope:** Assign exactly one leaf `category_id`, extract required
attributes, validate them against the category spec.

#### FR-2a Classification

- Candidate generation (deterministic): BM25/keyword match of
  `clean_title` + bullets against category `name` + ancestor path, top-K
  (default K = 8) leaf categories.
- LLM classification: prompt contains the cleaned listing and the K
  candidates only (ids + full path). The LLM must return
  `LLMClassification{category_id: Literal[candidates], confidence: float,
  rationale: str}` via structured output. A `category_id` not in the
  candidate set is a schema violation → retry → quarantine
  (`CLASSIFICATION_OUT_OF_SET`).
- If the deterministic top-1 score exceeds `HIGH_CONF_THRESHOLD` and the
  LLM agrees, mark `classification_source = "agree"`; if they disagree,
  record both and apply `CLASSIFICATION_DISAGREE` (configurable: warning
  or quarantine; default quarantine when LLM confidence < 0.8).

#### FR-2b Attribute extraction

- Deterministic extraction (FR-1 outputs) is the primary source.
- LLM extraction fills only attributes still missing, constrained to the
  category's `required_attributes_json`: the JSON schema sent to the LLM is
  **generated from the attribute spec** (enum → `enum`, int → `integer` with
  `minimum/maximum`, etc.), so an out-of-vocabulary value cannot parse.
- Reconciliation: deterministic value wins on conflict; conflict recorded
  as `ATTR_CONFLICT:<key>` (warning).

#### FR-2c Attribute validation (deterministic, Python only)

For each attribute spec of the assigned category:

| check | reason code | severity |
|---|---|---|
| required key missing | `ATTR_MISSING:<key>` | blocking |
| enum value not in `allowed` | `ATTR_ENUM_INVALID:<key>` | blocking |
| int/float outside `min`/`max` | `ATTR_RANGE:<key>` | blocking |
| string exceeds `max_len` | `ATTR_TOO_LONG:<key>` | blocking |
| `multi: false` but list given | `ATTR_MULTI_NOT_ALLOWED:<key>` | blocking |
| `pattern` mismatch | `ATTR_PATTERN:<key>` | blocking |
| attribute present but not required by category | `ATTR_EXTRA:<key>` | info (kept under `extra_attributes`) |

Acceptance criteria:

- AC-2.1 For the sample row, the assigned category is the shorts leaf
  under men's apparel and `gender == "men"`, `size == ["L"]`,
  `color == ["black","navy"]`, `pack_count == 2` with no blocking codes.
- AC-2.2 An LLM response that fails Pydantic validation is retried at most
  `LLM_MAX_RETRIES` (default 2) times with the validation error appended to
  the prompt; after exhaustion the item is quarantined with
  `LLM_SCHEMA_INVALID` and the raw responses are stored. The stage
  (`classify` / `extract`) goes in the reason's `detail`. Exhaustion where
  every parseable attempt named an id outside the shortlist is
  `CLASSIFICATION_OUT_OF_SET`; exhaustion caused only by transport failures
  (timeouts, 5xx, refusals) is `LLM_TRANSPORT_ERROR`. All three are blocking.
- AC-2.3 A non-leaf `category_id` can never be assigned (enforced by the
  `Literal` candidate set and a second explicit leaf check).
- AC-2.4 Changing `required_attributes_json` for a category and re-running
  validation (without re-running the LLM) re-evaluates outcomes from stored
  `CandidateListing` data. Implemented as
  `python -m catalog_onboard revalidate --db … [--limit N] [--dry-run]`:
  every draft and quarantine row stores its candidate; revalidate keeps the
  upstream (clean / candidates / classify) decisions, reuses the LLM
  attributes from the stored successful extract attempt, and re-runs
  reconcile → validate → comply → route with **zero** LLM calls. The
  outcome version is a hash of the rule files **and** the taxonomy content,
  so a spec change is a new version: `ingest` supersedes with fresh LLM
  calls, `revalidate` supersedes from stored data, and either is a no-op
  when re-run at the same version. A newly required attribute the LLM was
  never asked for becomes `ATTR_MISSING` (re-ingest to fill it).
- AC-2.5 Validation logic has 100 % branch coverage over the table above
  with table-driven unit tests; no LLM client is imported by the validation
  module.
- AC-2.6 The LLM never sees `prohibited_keywords` and is never asked a
  yes/no compliance question. Grep-level test: the prompt builder output
  contains no compliance vocabulary.

### FR-3 Draft catalog vs. quarantine pipeline

**Scope:** Route each `CandidateListing` based on `ValidationResult`.

Routing rule (deterministic):

```
blocking_codes == []  and compliance_ok  → draft_catalog (status=draft)
otherwise                                → quarantine   (status=quarantined)
```

Compliance checks (all deterministic, all run even after a first failure so
the full code list is captured):

| check | reason code |
|---|---|
| any `prohibited_keywords` of the assigned category or **any ancestor** matches normalized title/bullets (word-boundary, case-insensitive) | `PROHIBITED_KEYWORD:<kw>` |
| global prohibited list (`rules/global_prohibited.yaml`) matches | `PROHIBITED_GLOBAL:<kw>` |
| price missing or ≤ 0 | `PRICE_INVALID` |
| `clean_title` empty or < 3 chars | `TITLE_TOO_SHORT` |
| duplicate `(supplier_id, normalized fingerprint)` already in draft/live | `DUPLICATE_LISTING` |

Tables:

- `draft_catalog(listing_id PK, feed_item_id UNIQUE, supplier_id,
  category_id, clean_title, attributes_json, price, pipeline_run_id,
  rules_version, created_at)`
- `quarantine(quarantine_id PK, feed_item_id UNIQUE, supplier_id,
  proposed_category_id NULL, candidate_json, reason_codes_json,
  pipeline_run_id, rules_version, created_at, resolved_at NULL,
  resolution NULL)`
- `pipeline_runs(run_id PK, feed_item_id, stage, input_json, output_json,
  llm_raw_json NULL, duration_ms, created_at)`

Acceptance criteria:

- AC-3.1 A listing with any blocking reason code never appears in
  `draft_catalog`; a listing with zero blocking codes never appears in
  `quarantine`. Property test over randomly generated `ValidationResult`s.
- AC-3.2 Quarantined rows store **all** reason codes, not just the first.
- AC-3.3 Prohibited keyword matching is word-bounded: category keyword
  `knife` does not match `knifeless`; `gun` does not match `Gunnar`.
  Ancestor keywords are inherited.
- AC-3.4 Re-running the pipeline on an already-processed `feed_item_id`
  with the same `rules_version` is a no-op (0 new rows, exit code 0).
  Re-running with a new `rules_version` supersedes the previous outcome
  and retains the old one in `pipeline_runs`.
- AC-3.5 A single bad row (exception in any stage) is quarantined with
  `PIPELINE_ERROR:<stage>` and does not abort the batch.
- AC-3.6 CLI: `python -m catalog_onboard ingest --db path.sqlite
  [--limit N] [--supplier X] [--dry-run]` prints a summary table
  (processed / draft / quarantined / errors) and exits non-zero only on
  configuration errors.
- AC-3.7 The LLM client is behind an interface; `--llm fake` runs the
  whole pipeline with a deterministic fake for CI.

### FR-4 Catalog inspection API

**Scope:** Read-only FastAPI service over the SQLite DB.

| method | path | returns |
|---|---|---|
| GET | `/health` | `{status, rules_version, db_path}` |
| GET | `/catalog/draft` | paginated `DraftListing[]`; filters `supplier_id`, `category_id`, `q` (title substring) |
| GET | `/catalog/draft/{listing_id}` | `DraftListing` + full reason trail |
| GET | `/quarantine` | paginated `QuarantineItem[]`; filters `supplier_id`, `reason_code`, `proposed_category_id` |
| GET | `/quarantine/{quarantine_id}` | `QuarantineItem` + candidate + all stage outputs |
| GET | `/items/{feed_item_id}/trace` | ordered `pipeline_runs` for the item, LLM raw included |
| GET | `/taxonomy` | category tree with leaf flags and attribute specs |
| GET | `/taxonomy/{category_id}` | one node + effective (inherited) prohibited keywords |
| GET | `/stats` | counts by status, top reason codes, per-supplier quarantine rate |

Acceptance criteria:

- AC-4.1 All responses are Pydantic response models; OpenAPI at `/docs`
  validates with no `Any`-typed fields.
- AC-4.2 Pagination: `?limit=` (default 50, max 500) and `?cursor=`; stable
  ordering by `created_at, id`.
- AC-4.3 Unknown ids return 404 with `{detail, code}`; bad filters return
  422 (FastAPI default).
- AC-4.4 The API performs no writes and opens the DB read-only
  (`mode=ro` URI). Test: attempting any mutation raises.
- AC-4.5 `/items/{id}/trace` for a quarantined item includes the raw LLM
  responses that failed validation, so operators can debug prompts.
- AC-4.6 p95 latency for list endpoints < 100 ms on 10k rows (local
  SQLite, indexes on `supplier_id`, `category_id`, `created_at`).

## 6. Pydantic models (core)

```python
class AttributeSpec(BaseModel):
    key: str
    type: Literal["enum", "int", "float", "string", "bool"]
    allowed: list[str] | None = None
    min: float | None = None
    max: float | None = None
    max_len: int | None = None
    multi: bool = False
    pattern: str | None = None
    model_config = ConfigDict(extra="forbid")


class NormalizedListing(BaseModel):
    feed_item_id: str
    supplier_id: str
    clean_title: str
    bullets: list[str]
    tags: dict[str, str]
    unmapped_tags: dict[str, str]
    det_attrs: dict[str, list[str] | str | int | float | bool]
    keyword_tail: list[str]
    price_min: Decimal | None
    price_max: Decimal | None
    reason_codes: list[ReasonCode]
    raw_json: dict


class LLMClassification(BaseModel):  # Literal built per-call from candidates
    category_id: str
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(max_length=500)


class CandidateListing(BaseModel):
    normalized: NormalizedListing
    category_id: str
    classification_source: Literal["agree", "llm", "deterministic"]
    attributes: dict[str, Any]  # already type-coerced per spec
    extra_attributes: dict[str, Any]
    llm_trace: list[LLMAttempt]


class ReasonCode(BaseModel):
    code: str  # e.g. "ATTR_MISSING:size"
    severity: Literal["blocking", "warning", "info"]
    stage: Literal["clean", "extract", "classify", "validate", "comply", "route"]
    detail: str | None = None


class ValidationResult(BaseModel):
    reason_codes: list[ReasonCode]

    @property
    def blocking(self) -> list[ReasonCode]: ...
    @property
    def ok(self) -> bool:
        return not self.blocking
```

## 7. Deterministic vs. LLM boundary (explicit)

| concern | owner |
|---|---|
| Text cleaning, abbreviation expansion, tag mapping | Python |
| Size / color / pack / price parsing | Python regex + dictionaries |
| Candidate category shortlist | Python (BM25) |
| Pick category among shortlist | LLM, structured output, Literal-constrained |
| Fill missing required attributes | LLM, schema generated from spec |
| Attribute type/range/enum validation | Python |
| Prohibited keywords, price, title, duplicate checks | Python |
| Draft vs. quarantine routing | Python |
| "Is this compliant?" | **never asked of the LLM** |

## 8. Non-functional requirements

- Python 3.11, Pydantic v2, FastAPI, `sqlite3` stdlib (no ORM required).
- Config via env / `.env`: `LLM_PROVIDER` (`anthropic|fake`), model id
  (default latest Claude Sonnet-class), `LLM_MAX_RETRIES`,
  `HIGH_CONF_THRESHOLD`, `RULES_DIR`.
- Rules (`abbreviations.yaml`, `supplier_tag_map.yaml`,
  `global_prohibited.yaml`) are hashed into `rules_version`.
- Batch throughput target: ≥ 200 items/min with concurrency 8 against a
  real LLM; LLM calls bounded to ≤ 2 per item on the happy path
  (classify + extract) and may be merged into one.
- Structured logging (JSON) with `feed_item_id` and `run_id` on every line.
- No network access in the test suite.

## 9. Testing

| layer | what | tool |
|---|---|---|
| unit | cleaning rules, each extractor, each validation branch, compliance matcher | pytest, parametrized |
| snapshot | 50+ fixture rows → `NormalizedListing` JSON | pytest + syrupy or inline golden files |
| property | routing invariant (AC-3.1), word-boundary matcher | hypothesis |
| contract | LLM schema generation from `AttributeSpec`; invalid completion → retry → quarantine | pytest with fake client scripted responses |
| integration | full `ingest --llm fake` against starter DB, then API reads | pytest + httpx `TestClient` |
| regression | sample title from the brief end-to-end | pytest |

Definition of done: `pytest -q` green, `ruff` clean, `mypy --strict` on
`catalog_onboard/`, sample DB ingests with 0 `PIPELINE_ERROR` codes.

## 10. Out of scope (v1)

- Human review UI / quarantine resolution workflow (API is read-only).
- Publishing from draft to live catalog.
- Image processing, pricing strategy, inventory sync.
- Multi-language titles (English only; titles with letters outside the Latin script get a
  `LANG_UNSUPPORTED` warning, not quarantine. Accented Latin such as "Café" is accepted:
  a literal non-ASCII rule would flag ordinary brand names).
- Learning/updating the taxonomy itself.

## 11. Open questions (resolved in v1.1)

1. `prohibited_keywords` delimiter: **both** comma and pipe are accepted by the
   parser (`taxonomy.py`).
2. `CLASSIFICATION_DISAGREE`: **blocking** only when the deterministic top-1
   is confident (≥ `HIGH_CONF_THRESHOLD`) **and** LLM confidence <
   `DISAGREE_QUARANTINE_BELOW` (0.8); otherwise a warning and the LLM pick
   wins. `classification_source = "agree"` likewise requires a confident
   deterministic top-1; an LLM pick that matches a weak top-1 is `"llm"`.
3. Multi-variant listings: **one draft row with list attributes**; exploding
   into variants remains a follow-up.
4. Global prohibited list: `rules/global_prohibited.yaml` exists and is
   hashed into `rules_version`.
