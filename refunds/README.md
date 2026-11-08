# Automatic refunds (interview exercise)

Refund requests arrive against existing orders. The system decides, with no human in the loop,
whether to refund, how much and why, and records the result so the same money is never paid twice.

```bash
pytest tests/refunds -q                 # 56 tests, < 1 s
uvicorn refunds.api:app --reload        # demo orders o_1..o_4
curl -s -X POST localhost:8000/refund-requests -H 'content-type: application/json' \
  -d '{"request_id":"req-1","order_id":"o_1","amount":"40.00","reason":"damaged"}'
# send it again: identical body, still one refund
curl -s localhost:8000/orders/o_1/refunds
```

## Stack

| Part | What it uses | Why |
|---|---|---|
| Language | Python 3.9+, standard library for the whole domain | the decision, ledger and money code import nothing third-party, so they are trivially testable and portable |
| Money | `decimal.Decimal` quantised to cents | exact partial sums; floats are rejected at construction |
| Domain model | `dataclasses` (frozen) + `enum` | immutable values, no framework types in the core |
| Policy config | `pydantic` v2 model, frozen, versioned | validation on load from dict/JSON, immutable once active |
| Storage | in-memory `RefundLedger`, `IdempotencyStore`, `OrderRepo` | stand-ins for database tables; the SQL equivalent is in the `ledger.py` docstring |
| Concurrency | `threading.Lock` + per-order version (compare-and-set) | the lock plays the DB's row atomicity; CAS is what survives several instances |
| Payment provider | `PaymentProvider` protocol + `FakeProvider` (ok / decline / timeout) | the real SDK plugs in behind the same two calls, `refund` and `lookup` |
| API | FastAPI + pydantic request models | thin HTTP layer; amounts arrive as decimal strings, never floats |
| Tests | pytest, FastAPI `TestClient` (httpx) | 56 tests, under a second |

Install: `pip install -e ".[dev]"` from the repo root (same `pyproject.toml` as the shopfeed prototype; refunds needs only `fastapi`, `pydantic`, `uvicorn` and the dev extras).

## Invariants and where they live

| Invariant | Enforced in | How |
|---|---|---|
| total refunded for an order ≤ amount paid | `ledger.py::RefundLedger.reserve` — **only here** | `settled + reserved + amount ≤ paid`, checked and written atomically; PENDING and UNKNOWN count as reserved |
| a retried request never refunds twice | `idempotency.py` + `ledger.py` | replay by client `request_id`; the ledger also keys reservations by `request_id`, so a lost replay row still cannot reserve twice; the provider gets the same key |

`policy.py::cap_to_remaining` *also* looks at what is left, but only as a courtesy to produce a
sensible partial amount. A bug in any rule cannot over-pay: the ledger says no.

## Request path

```
RefundRequest ─► idempotency.get ─► load Order + LedgerView ─► decide() ─► ledger.reserve ─► provider.refund ─► settle/fail/unknown ─► idempotency.put
                 (replay or 409)     (signals: history, score)   PURE         CAS on version     key = request_id
```

Write order is deliberate: the reservation is written **before** the provider call (a crash after
paying leaves the money held, never free); the replay row is written **last** (a crash before it
makes the retry re-decide, and the ledger hands back the existing reservation).

## Domain model (`model.py`)

* `Order(paid_total, status, paid_at, delivered_at)` — `paid_total` includes shipping and tax; that is the ceiling.
* `RefundRequest(request_id, order_id, amount, reason, submitted_at)` — `request_id` is the client's idempotency key; `fingerprint()` = order + amount + reason.
* `Refund(status: PENDING | SETTLED | FAILED | UNKNOWN, policy_version, provider_ref, reasons)`.
* `Decision(outcome: APPROVE | ESCALATE | REJECT, amount, reasons, policy_version)`.
* `LedgerView(paid, settled, reserved, version)` — the snapshot the pure decision reads.
* `Money` (`money.py`) — `Decimal` quantised to cents, half-even, same-currency checked, **floats rejected**. 3 × 33.33 = 99.99 exactly.

## Rules (`policy.py`)

Each rule is `Context -> Optional[Verdict]`; a verdict carries a machine-readable reason. Evaluated
in order; REJECT stops (the request is malformed), ESCALATE collects every failing reason for the
reviewer, an APPROVE verdict with an amount adjusts the effective amount for later rules.

| Rule | Default | Fails as |
|---|---|---|
| `order_is_refundable` | status ∈ {paid, shipped, delivered} | REJECT `order_not_refundable:status=…` |
| `amount_is_valid` | > 0, same currency | REJECT `amount_invalid` / `currency_mismatch` |
| `cap_to_remaining` | ≤ paid − settled − reserved | cap → `capped_to_remaining:…`; nothing left → REJECT |
| `within_window` | ≤ 30 d since delivery (else since payment) | ESCALATE `outside_window:30d` |
| `reason_is_automatic` | `other` never; `changed_mind` only before shipment | ESCALATE `reason_requires_review:…` |
| `under_auto_limit` | ≤ 200.00 | ESCALATE `above_auto_limit:…` |
| `history_under_threshold` | < 3 refunds / 90 d per customer | ESCALATE `history_threshold:…` |
| `abuse_score_bounded` | ≥ 0.8 escalate; ≥ 0.5 halves the auto limit; no score → no-op | ESCALATE `abuse_score…` |

Adding a rule = one function + one line in `DEFAULT_RULES` + its tests. Thresholds are data in
`config.py::PolicyConfig`, which is immutable and versioned.

Decisions: unqualified requests are **escalated** (queued for a person), not rejected; REJECT is
reserved for impossible requests. Partial refunds are by amount; line-item refunds would add a
`lines` field to the request and a rule that sums line prices — the ledger does not change.

## Follow-ups, as built

1. **Several instances, shared DB.** Accounts carry a `version`; `reserve(expected_version=)` is a
   compare-and-set, and the service re-decides on `ConcurrencyConflict` (so the loser is *capped*,
   not rejected). SQL equivalent in the `ledger.py` docstring: one conditional `UPDATE … WHERE
   version = :v AND settled + reserved + :amt <= paid`, plus `UNIQUE(refunds.request_id)` as the
   second line of defence. `tests/refunds/test_concurrency.py` races threads and two service
   instances on one ledger.
2. **Provider timeout.** Refund stays `UNKNOWN`, money stays held. `reconcile.py` looks our
   `request_id` up at the provider: found → SETTLED; not found → re-issue under the same key
   (provider idempotency makes a late duplicate impossible); declined → FAILED and released.
   Never release on "not found" alone — the first call may still be in flight.
3. **Abuse score.** One rule, fed through `Signals.abuse_score`. It can escalate or shrink the
   ceiling, never approve; missing score is a no-op; the reason string carries score and threshold.
   Bounded means: the fixed rules are the outer envelope, the model only narrows it.
4. **Thresholds without a deploy.** `RefundService.set_policy` / `PUT /admin/policy`; same version
   with different content is refused; every `Decision` and `Refund` stores `policy_version`; a
   retry replays the original decision even after the policy changed.

## Edge cases covered by tests

Three partials summing to the cent; pending/unknown holds blocking a second payout; failed refunds
releasing their hold; capped amount dropping under the auto limit; window anchored on delivery vs
payment, boundary inclusive; `changed_mind` after shipment; float amounts rejected at the API;
reused `request_id` with a new payload → 409; idempotency row lost after payment; two instances
racing on one order; timeout with and without the money having moved; policy change mid-history.

## Not modelled (would say so in the interview)

Persistence (in-memory ledger stands in for a table), line-item refunds, multi-currency orders,
customer notification, a real provider SDK, auth on `/admin`.
