"""Run lifecycle: extract -> recommend inside one repricing_runs record."""

from __future__ import annotations

import sqlite3
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from repricer.db import repo
from repricer.db.connection import transaction
from repricer.engine.evaluate import evaluate
from repricer.engine.policy import PolicyStore

ExtractFn = Callable[[sqlite3.Connection, str], int]


@dataclass(frozen=True)
class RunResult:
    run_id: str
    skus_considered: int
    recommendations: int
    offers_extracted: int
    status: str
    error: str | None = None


def recommend_stage(
    conn: sqlite3.Connection,
    run_id: str,
    store: PolicyStore,
    now: datetime,
    hard_cap_bp: int,
) -> tuple[int, int]:
    """Evaluate every SKU. Expires earlier pending recs for the SKU. Returns (skus, recs)."""
    skus = 0
    recs = 0
    with transaction(conn):
        for inv in repo.list_inventory(conn):
            policy = store.for_class(inv.price_class)
            offers = repo.eligible_offers(conn, inv.sku, policy, now)
            rec = evaluate(inv, offers, policy, now, hard_cap_bp=hard_cap_bp)
            repo.expire_pending_for_sku(conn, inv.sku)
            repo.insert_recommendation(
                conn, rec, run_id=run_id, policy_version=store.version, now=now
            )
            skus += 1
            recs += 1
    return skus, recs


def run(
    conn: sqlite3.Connection,
    *,
    store: PolicyStore,
    trigger: str,
    now: datetime,
    hard_cap_bp: int,
    extract: ExtractFn | None = None,
    stages: tuple[str, ...] = ("extract", "recommend"),
) -> RunResult:
    with transaction(conn):
        repo.replace_policies(conn, store.policies, store.version, now)
        run_id = repo.insert_run(conn, trigger=trigger, policy_version=store.version, now=now)
    offers = 0
    skus = 0
    recs = 0
    try:
        if "extract" in stages and extract is not None:
            offers = extract(conn, run_id)
        if "recommend" in stages:
            skus, recs = recommend_stage(conn, run_id, store, now, hard_cap_bp)
    except Exception as e:  # noqa: BLE001 - a run must always be closed out
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        with transaction(conn):
            repo.finish_run(
                conn,
                run_id,
                status="failed",
                now=now,
                skus_considered=skus,
                recommendations=recs,
                offers_extracted=offers,
                error=err,
            )
        return RunResult(run_id, skus, recs, offers, "failed", err)
    with transaction(conn):
        repo.finish_run(
            conn,
            run_id,
            status="completed",
            now=now,
            skus_considered=skus,
            recommendations=recs,
            offers_extracted=offers,
        )
    return RunResult(run_id, skus, recs, offers, "completed")
