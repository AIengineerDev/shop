"""Orchestration. Everything stateful happens here, in a deliberate order:

  1. replay check          get_refund(request_id); same fingerprint -> return stored result, else 409
  2. read                  order, ledger view, signals (customer history, abuse score)
  3. decide                pure
  4. write the decision    REJECT/ESCALATE -> record_decision (holds nothing)
                           APPROVE         -> ledger.reserve (holds the money; THE invariant check)
  5. call the provider     only after the hold is committed
  6. settle / fail / unknown

Why reserve before calling: a crash between 5 and 6 leaves the money *held* (UNKNOWN), never free. A
retry then hits step 1 and replays. If a lost process means the row is PENDING forever, the
reconciler treats stale PENDING like UNKNOWN. The reservation losing a race (InsufficientFunds)
re-runs the decision once against the fresh ledger view, so the slower request is capped to what is
left rather than rejected.
"""
from __future__ import annotations

import json
import sqlite3
from typing import List, Optional, Tuple

from refundsv2.ledger import (DuplicateRequest, InsufficientFunds, UnknownOrder, count_customer_refunds_since, get_order, get_refund,
                              ledger_view, list_by_status, record_decision, reserve, transition)
from refundsv2.model import Outcome, Refund, RefundRequest, RefundStatus, Signals
from refundsv2.policy import DAY, DEFAULT_POLICY, PolicyConfig, decide
from refundsv2.provider import PaymentProvider, ProviderTimeout


MAX_RESERVE_ATTEMPTS = 5


class DuplicateRequestConflict(Exception):
    """Same request_id, different payload."""


class PolicyVersionConflict(Exception):
    pass


class RefundService:
    def __init__(self, conn: sqlite3.Connection, provider: PaymentProvider, policy: PolicyConfig = DEFAULT_POLICY,
                 now: float = 0.0, grace_s: float = 300.0) -> None:
        self.conn = conn
        self.provider = provider
        self.grace_s = grace_s
        self.policy = self._load_active_policy() or self._store_policy(policy, now)

    # ------------------------------------------------------------------ policy
    def _load_active_policy(self) -> Optional[PolicyConfig]:
        r = self.conn.execute("SELECT config_json FROM policies WHERE active = 1").fetchone()
        return PolicyConfig.model_validate_json(r["config_json"]) if r else None

    def _store_policy(self, cfg: PolicyConfig, now: float) -> PolicyConfig:
        self.conn.execute("BEGIN IMMEDIATE")
        self.conn.execute("UPDATE policies SET active = 0")
        self.conn.execute("INSERT INTO policies (version, config_json, activated_at, active) VALUES (?,?,?,1)", (cfg.version, cfg.model_dump_json(), now))
        self.conn.execute("COMMIT")
        return cfg

    def set_policy(self, cfg: PolicyConfig, now: float) -> PolicyConfig:
        """Operations change thresholds here. A version is immutable: same version, different content -> conflict."""
        r = self.conn.execute("SELECT config_json FROM policies WHERE version = ?", (cfg.version,)).fetchone()
        if r is not None:
            if json.loads(r["config_json"]) != json.loads(cfg.model_dump_json()):
                raise PolicyVersionConflict(f"policy {cfg.version} already exists with different content")
            self.conn.execute("BEGIN IMMEDIATE"); self.conn.execute("UPDATE policies SET active = (version = ?)", (cfg.version,)); self.conn.execute("COMMIT")
            self.policy = cfg
            return cfg
        self.policy = self._store_policy(cfg, now)
        return cfg

    def refresh_policy(self) -> PolicyConfig:
        """Other instances pick up a change by re-reading the active row."""
        self.policy = self._load_active_policy() or self.policy
        return self.policy

    # ------------------------------------------------------------------ request path
    def handle(self, req: RefundRequest, now: float, abuse_score: Optional[float] = None) -> Refund:
        existing = get_refund(self.conn, req.request_id)
        if existing is not None:
            return self._replay(existing, req)
        order = get_order(self.conn, req.order_id)
        if order is None:
            raise UnknownOrder(req.order_id)
        signals = Signals(refunds_last_90d=count_customer_refunds_since(self.conn, order.customer_id, now - 90 * DAY), abuse_score=abuse_score)

        for attempt in range(MAX_RESERVE_ATTEMPTS):  # re-decide only after losing a reservation race
            decision = decide(order, req, ledger_view(self.conn, req.order_id), signals, self.policy, now)
            try:
                if decision.outcome == Outcome.APPROVE:
                    refund = reserve(self.conn, req, decision, now)
                    break
                status = RefundStatus.ESCALATED if decision.outcome == Outcome.ESCALATE else RefundStatus.REJECTED
                return record_decision(self.conn, req, decision, status, now)
            except DuplicateRequest:
                return self._replay(get_refund(self.conn, req.request_id), req)  # same id arrived concurrently
            except InsufficientFunds:
                if attempt == MAX_RESERVE_ATTEMPTS - 1:
                    raise  # pathological contention: tell the caller to retry; nothing was written
        return self._execute(refund, order.currency, now)

    def _replay(self, existing: Refund, req: RefundRequest) -> Refund:
        if existing.fingerprint != req.fingerprint():
            raise DuplicateRequestConflict(req.request_id)
        return existing

    def _execute(self, refund: Refund, currency: str, now: float) -> Refund:
        """Provider call after the hold is committed; map the answer onto the row."""
        try:
            res = self.provider.refund(refund.request_id, refund.amount_cents, currency)
        except ProviderTimeout:
            transition(self.conn, refund.request_id, [RefundStatus.PENDING, RefundStatus.UNKNOWN], RefundStatus.UNKNOWN, now, bump_attempts=True)
            return get_refund(self.conn, refund.request_id)
        self._apply(refund.request_id, res.status, res.ref, now)
        return get_refund(self.conn, refund.request_id)

    def _apply(self, request_id: str, status: str, ref: Optional[str], now: float) -> None:
        froms = [RefundStatus.PENDING, RefundStatus.UNKNOWN]
        if status == "ok":
            transition(self.conn, request_id, froms, RefundStatus.SETTLED, now, provider_ref=ref, bump_attempts=True)
        else:
            transition(self.conn, request_id, froms, RefundStatus.FAILED, now, bump_attempts=True)

    # ------------------------------------------------------------------ reconciliation
    def reconcile(self, now: float) -> List[Tuple[str, str]]:
        """For every UNKNOWN (and stale PENDING) refund, ask the provider what happened.

        found ok        -> SETTLED            (money moved; our record catches up)
        found declined  -> FAILED             (money released)
        not found       -> leave it held until the grace period passes, then re-issue under the SAME
                           request_id. Never release on "not found" alone: the first call may still land.
        """
        actions = []
        stale = [r for r in list_by_status(self.conn, RefundStatus.PENDING) if now - r.updated_at >= self.grace_s]
        for r in list_by_status(self.conn, RefundStatus.UNKNOWN) + stale:
            order = get_order(self.conn, r.order_id)
            res = self.provider.lookup(r.request_id)
            if res is not None:
                self._apply(r.request_id, res.status, res.ref, now)
                actions.append((r.request_id, "settled" if res.status == "ok" else "failed"))
            elif now - r.updated_at >= self.grace_s:
                self._execute(r, order.currency, now)
                actions.append((r.request_id, "reissued"))
            else:
                actions.append((r.request_id, "waiting"))
        return actions
