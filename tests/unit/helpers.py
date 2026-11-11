"""Shared test helpers for execution-layer tests."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from repricer.db import repo
from repricer.db.connection import transaction
from repricer.engine.evaluate import evaluate
from repricer.models import EligibleOffer, InventorySnapshot, Policy, Strategy

POLICY = Policy(
    strategy=Strategy.UNDERCUT_LOWEST,
    undercut_cents=50,
    undercut_bp=100,
    max_decrease_bp=1000,
    max_increase_bp=500,
    min_competitor_confidence=60,
    max_offer_age_hours=72,
    low_stock_threshold=3,
    round_to_cents=99,
)


def add_inventory(
    conn: sqlite3.Connection,
    sku: str,
    now: datetime,
    *,
    cost: int = 3000,
    margin_bp: int = 2000,
    current: int = 4500,
    stock: int = 25,
) -> InventorySnapshot:
    inv = InventorySnapshot(
        sku=sku,
        title=f"title {sku}",
        cost_price_cents=cost,
        min_margin_bp=margin_bp,
        current_price_cents=current,
        stock_count=stock,
        price_class="default",
    )
    with transaction(conn):
        repo.insert_inventory(conn, inv, now)
    return inv


def new_run(conn: sqlite3.Connection, now: datetime) -> str:
    with transaction(conn):
        return repo.insert_run(conn, trigger="test", policy_version=1, now=now)


def recommend(
    conn: sqlite3.Connection,
    sku: str,
    now: datetime,
    *,
    competitor_prices: tuple[int, ...] = (3799, 4449),
    policy: Policy = POLICY,
    run_id: str | None = None,
) -> str:
    """Evaluate against fabricated in-memory offers and persist as pending."""
    inv = repo.get_inventory(conn, sku)
    offers = [
        EligibleOffer(
            offer_id=i + 1,
            competitor="c",
            effective_price_cents=p,
            confidence=85,
            scraped_at=now - timedelta(hours=1),
        )
        for i, p in enumerate(competitor_prices)
    ]
    rec = evaluate(inv, offers, policy, now)
    rid = run_id or new_run(conn, now)
    with transaction(conn):
        return repo.insert_recommendation(conn, rec, run_id=rid, policy_version=1, now=now)
