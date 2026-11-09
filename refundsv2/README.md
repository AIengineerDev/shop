# Automatic refunds, second build

Refund requests arrive against orders. The system decides with no human in the loop whether to
refund, how much and why, and records the result so the same money is never paid twice.

```bash
pip install -e ".[dev]"
pytest tests/refundsv2 -q                      # 79 tests, under a second
uvicorn refundsv2.api:app --reload             # demo orders o_1 .. o_4, in-memory database
```

```bash
curl -s -X POST localhost:8000/refund-requests -H 'content-type: application/json' \
  -d '{"request_id":"req-1","order_id":"o_1","amount":"40.00","reason":"damaged"}'
```

Send the same body again: the response is identical, `GET /orders/o_1/refunds` shows one refund.

## How this build differs from the first one

| Concern | First build (`task/refunds-auto-refund-system`) | This build |
|---|---|---|
| Storage | in-memory, `threading.Lock` + per-order version | SQLite with real transactions; the ceiling is checked by the database |
| Ledger | mutable account (settled, reserved) | append-only refund rows; "remaining" is derived with `SUM`, never stored |
| Decision | rule functions returning verdicts | policy as data plus a rule registry; the decision carries a trace of every rule |
| Provider call | inside the request | after the hold is committed; `UNKNOWN` rows reconciled separately |
| Idempotency | separate store | `request_id` is the primary key of the refund row |
| Money | `Decimal` value type | integer cents; `Money` only formats |

## The two invariants and where they live

| Invariant | Enforced in | How |
|---|---|---|
| total refunded for an order never exceeds what was paid | `ledger.py::reserve`, **only here** | one `INSERT … SELECT … WHERE paid − SUM(held) ≥ amount` under `BEGIN IMMEDIATE`; held = PENDING, SETTLED, UNKNOWN |
| a retried request never refunds twice | `refunds.request_id PRIMARY KEY` + `service.py::_replay` | the row is the idempotency record; same id with a different payload is a 409; the provider receives the same key |

`policy.py::cap_to_remaining` also looks at what is left, but only to produce a sensible partial
amount. A bug in any rule cannot over-pay, because the database refuses the row.

## Request path

```
request ─► replay check ─► read order, ledger view, signals ─► decide() ─► record or reserve ─► provider ─► settle / fail / unknown
           by request_id                                       PURE         one conditional     same key
           same payload: return stored result                               INSERT
           different payload: 409
```

Write order matters. The hold is committed **before** the provider is called, so a crash after the
money moved leaves the row PENDING or UNKNOWN and the money held, never free. A retry hits the
replay check and returns the stored row. A reservation that loses a race raises
`InsufficientFunds`; the service re-decides against the fresh ledger, so the slower request is
capped to what is left instead of rejected.

## Domain model (`model.py`)

`Order(paid_cents, currency, status, paid_at, shipped_at, delivered_at)`: `paid_cents` includes
shipping and tax and is the ceiling. `RefundRequest(request_id, order_id, amount_cents, currency,
reason, submitted_at)`: `fingerprint()` covers order, amount, currency and reason, not the
timestamp. `Decision(outcome, amount_cents, reasons, trace, policy_version)`. `Refund(status)` with
statuses PENDING, SETTLED, UNKNOWN (hold money) and FAILED, ESCALATED, REJECTED (do not).
`LedgerView(paid, held)` is what the pure decision reads. `Signals(refunds_last_90d, abuse_score)`
are the inputs that come from neither the order nor the ledger.

## Rules (`policy.py`)

Each rule is `Context → RuleResult(name, passed, reason, effect)`. Effects: `reject` stops
(impossible request), `cap` lowers the amount, `limit` lowers the automatic ceiling, `escalate`
queues the request for a person and collects every failing reason. The default order:

| Rule | Default | On failure |
|---|---|---|
| `order_refundable` | status in paid, shipped, delivered | REJECT `order_not_refundable:status=…` |
| `amount_valid` | positive, same currency | REJECT |
| `cap_to_remaining` | amount ≤ paid − held | cap `capped_to_remaining:…`; nothing left → REJECT |
| `within_window` | ≤ 30 days since delivery, else since payment; inclusive | ESCALATE `outside_window:30d` |
| `reason_automatic` | damaged, not as described, late, not received; `changed_mind` only before shipment; `other` never | ESCALATE `reason_requires_review:…` |
| `history_threshold` | fewer than 3 held refunds per customer in 90 days | ESCALATE |
| `abuse_score_bounded` | ≥ 0.8 escalate; ≥ 0.5 halve the ceiling; absent → no effect | ESCALATE / `limit` |
| `under_auto_limit` | capped amount ≤ ceiling (200.00 by default) | ESCALATE `above_auto_limit:…` |

Adding a rule is one function plus one line in `RULES` plus its tests. Unqualified requests are
escalated, not rejected; REJECT is reserved for requests that can never succeed. Partial refunds
are by amount. Line-item refunds would add `lines` to the request and a rule that sums them; the
ledger does not change.

## Follow-ups

**Several instances behind a load balancer, shared database.** Nothing in process memory
matters. The conditional insert runs under the database's write lock, so two instances cannot both
read "60 remaining" and both insert 60. `tests/refundsv2/test_concurrency.py` runs two and twelve
service instances, each with its own connection, against one file database; the total held never
exceeds the amount paid and the loser is capped. In PostgreSQL the same statement works with
`SELECT … FOR UPDATE` on the order row or a serializable transaction with retry; the primary key on
`request_id` is the second line of defence either way.

**Provider timeout after we decided to refund.** The row stays UNKNOWN and the money stays held.
`service.reconcile()` asks the provider what happened to our `request_id`: found and paid →
SETTLED; declined → FAILED and released; not found → leave it until a grace period passes, then
re-issue under the same key, which the provider treats idempotently. Never release on "not found"
alone: the first call may still be in flight. Stale PENDING rows (a crash between hold and call)
are treated the same way.

**Abuse-model score.** One rule, fed through `Signals.abuse_score`. It can escalate or shrink the
automatic ceiling, never approve, and a missing score is a no-op. The reason string carries the
score and the threshold it crossed, so every automatic decision stays explainable. The fixed rules
are the outer envelope; the model only narrows it.

**Thresholds without a deploy.** `PUT /admin/policy` stores a `PolicyConfig` in the `policies`
table and activates it. A version is immutable: the same version with different content is a 409.
Every decision and refund row records `policy_version`. Other instances call `refresh_policy()` (or
read the active row on start). A retry replays the original decision even after the policy changed.

## Edge cases the tests pin down

Three partials of 33.33 summing to 99.99 then a 500 request capped to 0.01 then nothing left;
PENDING and UNKNOWN holds blocking a second payout; FAILED releasing; the capped amount dropping
under the automatic limit; window inclusive at the boundary and anchored on delivery when present;
`changed_mind` before versus after shipment; float amounts rejected at the API; reused `request_id`
with a new payload; retry after a timeout paying once; provider paid but our lookup is empty and the
re-issue is idempotent; policy change between the original request and its retry; a crash between
the hold and the provider call.

## Not modelled

A real provider SDK, line-item refunds, multi-currency orders, customer notification, auth on
`/admin`, a scheduler for `reconcile` (it is a function; a cron or queue consumer calls it).
