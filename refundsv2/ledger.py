"""The ledger: the ONE place the over-refund invariant is enforced.

`reserve()` is a single conditional INSERT ... SELECT inside BEGIN IMMEDIATE:

    insert the refund row only if  paid - SUM(held rows for this order) >= amount

Held rows are PENDING, SETTLED and UNKNOWN. A FAILED row releases its money; ESCALATED and REJECTED
rows record a decision and hold nothing. Because the check and the write are one statement under the
database's write lock, it holds across threads, processes and instances sharing the database. The
PRIMARY KEY on request_id is the second invariant: a retry cannot insert a second row.

In PostgreSQL this is the same statement; `BEGIN IMMEDIATE` becomes `SELECT ... FOR UPDATE` on the
order row or a serializable transaction with retry.
"""
from __future__ import annotations

import sqlite3
from typing import Iterable, List, Optional

from refundsv2.model import (HOLDS, Decision, LedgerView, Order, OrderStatus, Refund, RefundRequest, RefundStatus)

HOLD_SQL = ",".join(f"'{s.value}'" for s in HOLDS)


class InsufficientFunds(Exception):
    """The conditional insert matched no row: amount exceeds what is left on the order."""


class DuplicateRequest(Exception):
    """A refund row with this request_id already exists."""


class UnknownOrder(Exception):
    pass


# ---------------------------------------------------------------- orders
def insert_order(conn: sqlite3.Connection, o: Order) -> None:
    conn.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)",
                 (o.order_id, o.customer_id, o.paid_cents, o.currency, o.status.value, o.paid_at, o.shipped_at, o.delivered_at))


def get_order(conn: sqlite3.Connection, order_id: str) -> Optional[Order]:
    r = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    if r is None:
        return None
    return Order(r["order_id"], r["customer_id"], r["paid_cents"], r["currency"], OrderStatus(r["status"]), r["paid_at"], r["shipped_at"], r["delivered_at"])


def ledger_view(conn: sqlite3.Connection, order_id: str) -> LedgerView:
    r = conn.execute(f"""
        SELECT o.paid_cents AS paid,
               COALESCE((SELECT SUM(amount_cents) FROM refunds WHERE order_id = o.order_id AND status IN ({HOLD_SQL})), 0) AS held
        FROM orders o WHERE o.order_id = ?""", (order_id,)).fetchone()
    if r is None:
        raise UnknownOrder(order_id)
    return LedgerView(paid_cents=r["paid"], held_cents=r["held"])


# ---------------------------------------------------------------- refunds
def _row_to_refund(r: sqlite3.Row) -> Refund:
    return Refund(r["request_id"], r["order_id"], r["amount_cents"], RefundStatus(r["status"]), Decision.from_json(r["decision_json"]),
                  r["fingerprint"], r["created_at"], r["updated_at"], r["provider_ref"], r["attempts"])


def get_refund(conn: sqlite3.Connection, request_id: str) -> Optional[Refund]:
    r = conn.execute("SELECT * FROM refunds WHERE request_id = ?", (request_id,)).fetchone()
    return _row_to_refund(r) if r else None


def list_refunds(conn: sqlite3.Connection, order_id: str) -> List[Refund]:
    return [_row_to_refund(r) for r in conn.execute("SELECT * FROM refunds WHERE order_id = ? ORDER BY created_at, request_id", (order_id,))]


def list_by_status(conn: sqlite3.Connection, status: RefundStatus) -> List[Refund]:
    return [_row_to_refund(r) for r in conn.execute("SELECT * FROM refunds WHERE status = ? ORDER BY created_at", (status.value,))]


def record_decision(conn: sqlite3.Connection, req: RefundRequest, decision: Decision, status: RefundStatus, now: float) -> Refund:
    """Store an ESCALATED or REJECTED decision (amount 0, holds nothing) so a retry replays it."""
    assert status in (RefundStatus.ESCALATED, RefundStatus.REJECTED)
    try:
        conn.execute("INSERT INTO refunds (request_id, order_id, amount_cents, status, fingerprint, decision_json, policy_version, created_at, updated_at) "
                     "VALUES (?,?,?,?,?,?,?,?,?)",
                     (req.request_id, req.order_id, 0, status.value, req.fingerprint(), decision.to_json(), decision.policy_version, now, now))
    except sqlite3.IntegrityError as e:
        raise DuplicateRequest(req.request_id) from e
    return get_refund(conn, req.request_id)


def reserve(conn: sqlite3.Connection, req: RefundRequest, decision: Decision, now: float) -> Refund:
    """Hold `decision.amount_cents` against the order, atomically checking the ceiling. THE invariant."""
    amount = decision.amount_cents
    if amount <= 0:
        raise ValueError("reserve needs a positive amount")
    conn.execute("BEGIN IMMEDIATE")
    try:
        cur = conn.execute(f"""
            INSERT INTO refunds (request_id, order_id, amount_cents, status, fingerprint, decision_json, policy_version, created_at, updated_at)
            SELECT ?, o.order_id, ?, '{RefundStatus.PENDING.value}', ?, ?, ?, ?, ?
            FROM orders o
            WHERE o.order_id = ?
              AND o.paid_cents
                  - COALESCE((SELECT SUM(r.amount_cents) FROM refunds r WHERE r.order_id = o.order_id AND r.status IN ({HOLD_SQL})), 0)
                  >= ?""",
            (req.request_id, amount, req.fingerprint(), decision.to_json(), decision.policy_version, now, now, req.order_id, amount))
        if cur.rowcount != 1:
            conn.execute("ROLLBACK")
            if get_order(conn, req.order_id) is None:
                raise UnknownOrder(req.order_id)
            raise InsufficientFunds(f"{req.order_id}: cannot hold {amount}")
        conn.execute("COMMIT")
    except sqlite3.IntegrityError as e:
        conn.execute("ROLLBACK")
        raise DuplicateRequest(req.request_id) from e
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    return get_refund(conn, req.request_id)


def transition(conn: sqlite3.Connection, request_id: str, from_statuses: Iterable[RefundStatus], to: RefundStatus, now: float,
               provider_ref: Optional[str] = None, bump_attempts: bool = False) -> bool:
    """Guarded status change. Returns False if the row was not in one of `from_statuses` (lost race / already done)."""
    froms = [s.value for s in from_statuses]
    cur = conn.execute(f"""
        UPDATE refunds SET status = ?, updated_at = ?,
               provider_ref = COALESCE(?, provider_ref),
               attempts = attempts + ?
        WHERE request_id = ? AND status IN ({",".join("?" * len(froms))})""",
        (to.value, now, provider_ref, 1 if bump_attempts else 0, request_id, *froms))
    return cur.rowcount == 1


def count_customer_refunds_since(conn: sqlite3.Connection, customer_id: str, since: float) -> int:
    r = conn.execute(f"""
        SELECT COUNT(*) AS n FROM refunds r JOIN orders o ON o.order_id = r.order_id
        WHERE o.customer_id = ? AND r.status IN ({HOLD_SQL}) AND r.created_at >= ?""", (customer_id, since)).fetchone()
    return int(r["n"])
