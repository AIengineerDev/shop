"""Typed query layer. The only module that writes SQL for domain tables.

Writes take an explicit connection and never open their own transaction: callers wrap
them with ``repricer.db.connection.transaction``.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any, cast

from repricer.clock import iso, parse_iso
from repricer.ids import new_ulid
from repricer.models import (
    EligibleOffer,
    InventorySnapshot,
    OfferStatus,
    Policy,
    Recommendation,
    ReviewStatus,
)


class NotFound(LookupError):
    pass


class ConcurrentUpdate(RuntimeError):
    pass


# --- inventory --------------------------------------------------------------


def get_inventory(conn: sqlite3.Connection, sku: str) -> InventorySnapshot:
    row = conn.execute("SELECT * FROM inventory WHERE sku = ?", (sku,)).fetchone()
    if row is None:
        raise NotFound(f"sku {sku!r}")
    return _inv(row)


def list_inventory(conn: sqlite3.Connection) -> list[InventorySnapshot]:
    return [_inv(r) for r in conn.execute("SELECT * FROM inventory ORDER BY sku")]


def list_inventory_titles(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    return [(r["sku"], r["title"]) for r in conn.execute("SELECT sku, title FROM inventory")]


def _inv(row: sqlite3.Row) -> InventorySnapshot:
    return InventorySnapshot(
        sku=row["sku"],
        title=row["title"],
        cost_price_cents=row["cost_price_cents"],
        min_margin_bp=row["min_margin_bp"],
        current_price_cents=row["current_price_cents"],
        stock_count=row["stock_count"],
        price_class=row["price_class"],
    )


def insert_inventory(conn: sqlite3.Connection, inv: InventorySnapshot, now: datetime) -> None:
    conn.execute(
        "INSERT INTO inventory (sku, title, cost_price_cents, min_margin_bp, "
        "current_price_cents, stock_count, price_class, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (
            inv.sku,
            inv.title,
            inv.cost_price_cents,
            inv.min_margin_bp,
            inv.current_price_cents,
            inv.stock_count,
            inv.price_class,
            iso(now),
            iso(now),
        ),
    )


def _set_inventory_price(
    conn: sqlite3.Connection, sku: str, expected_old: int, new: int, now: datetime
) -> None:
    """Private: only ``repricer.execution.apply.apply_change`` may call this."""
    cur = conn.execute(
        "UPDATE inventory SET current_price_cents = ?, updated_at = ? "
        "WHERE sku = ? AND current_price_cents = ?",
        (new, iso(now), sku, expected_old),
    )
    if cur.rowcount != 1:
        raise ConcurrentUpdate(f"inventory {sku!r} price changed underneath us")


# --- scrapes ----------------------------------------------------------------


def insert_scrape(
    conn: sqlite3.Connection,
    *,
    sku: str | None,
    competitor: str,
    raw_text: str,
    scraped_at: datetime,
    now: datetime,
) -> int:
    cur = conn.execute(
        "INSERT INTO competitor_scrapes (sku, competitor, raw_text, scraped_at, extracted, "
        "created_at) VALUES (?,?,?,?,0,?)",
        (sku, competitor, raw_text, iso(scraped_at), iso(now)),
    )
    return int(cur.lastrowid or 0)


def unextracted_scrapes(conn: sqlite3.Connection, limit: int | None = None) -> list[sqlite3.Row]:
    sql = "SELECT * FROM competitor_scrapes WHERE extracted = 0 ORDER BY scrape_id"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return list(conn.execute(sql))


def all_scrapes(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM competitor_scrapes ORDER BY scrape_id"))


def get_scrape(conn: sqlite3.Connection, scrape_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM competitor_scrapes WHERE scrape_id = ?", (scrape_id,)
    ).fetchone()
    if row is None:
        raise NotFound(f"scrape {scrape_id}")
    return cast(sqlite3.Row, row)


def mark_scrape_extracted(conn: sqlite3.Connection, scrape_id: int) -> None:
    conn.execute("UPDATE competitor_scrapes SET extracted = 1 WHERE scrape_id = ?", (scrape_id,))


# --- offers -----------------------------------------------------------------


def insert_offer(
    conn: sqlite3.Connection,
    *,
    scrape_id: int,
    sku: str | None,
    competitor: str,
    listed_price_cents: int | None,
    coupon_cents: int,
    coupon_bp: int,
    shipping_cents: int | None,
    effective_price_cents: int | None,
    currency: str,
    condition: str,
    in_stock: bool | None,
    scraped_at: datetime,
    extractor: str,
    confidence: int,
    status: OfferStatus,
    reject_reason: str | None,
    reason_codes: list[str],
    raw_llm_json: str | None,
    now: datetime,
) -> int:
    cur = conn.execute(
        "INSERT INTO competitor_offers (scrape_id, sku, competitor, listed_price_cents, "
        "coupon_cents, coupon_bp, shipping_cents, effective_price_cents, currency, condition, "
        "in_stock, scraped_at, extractor, confidence, status, reject_reason, reason_codes, "
        "raw_llm_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            scrape_id,
            sku,
            competitor,
            listed_price_cents,
            coupon_cents,
            coupon_bp,
            shipping_cents,
            effective_price_cents,
            currency,
            condition,
            None if in_stock is None else int(in_stock),
            iso(scraped_at),
            extractor,
            confidence,
            status.value,
            reject_reason,
            json.dumps(reason_codes),
            raw_llm_json,
            iso(now),
        ),
    )
    return int(cur.lastrowid or 0)


def supersede_offers_for_scrape(conn: sqlite3.Connection, scrape_id: int) -> int:
    cur = conn.execute(
        "UPDATE competitor_offers SET status = 'superseded' "
        "WHERE scrape_id = ? AND status != 'superseded'",
        (scrape_id,),
    )
    return int(cur.rowcount)


def eligible_offers(
    conn: sqlite3.Connection, sku: str, policy: Policy, now: datetime
) -> list[EligibleOffer]:
    """Spec §5.1: accepted, USD, new, confident, fresh, not known-out-of-stock."""
    cutoff = iso(now - timedelta(hours=policy.max_offer_age_hours))
    rows = conn.execute(
        "SELECT * FROM competitor_offers WHERE sku = ? AND status = 'accepted' "
        "AND currency = 'USD' AND condition = 'new' AND confidence >= ? "
        "AND scraped_at >= ? AND (in_stock IS NULL OR in_stock = 1) "
        "AND effective_price_cents IS NOT NULL ORDER BY effective_price_cents, offer_id",
        (sku, policy.min_competitor_confidence, cutoff),
    ).fetchall()
    out: list[EligibleOffer] = []
    for r in rows:
        codes = json.loads(r["reason_codes"] or "[]")
        out.append(
            EligibleOffer(
                offer_id=r["offer_id"],
                competitor=r["competitor"],
                effective_price_cents=r["effective_price_cents"],
                confidence=r["confidence"],
                scraped_at=parse_iso(r["scraped_at"]),
                shipping_unknown="SHIPPING_UNKNOWN" in codes,
            )
        )
    return out


def offers_for_sku(conn: sqlite3.Connection, sku: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT o.*, s.raw_text FROM competitor_offers o "
            "JOIN competitor_scrapes s ON s.scrape_id = o.scrape_id "
            "WHERE o.sku = ? ORDER BY o.offer_id",
            (sku,),
        )
    )


def offers_by_ids(conn: sqlite3.Connection, ids: list[int]) -> list[sqlite3.Row]:
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    return list(
        conn.execute(
            f"SELECT o.*, s.raw_text FROM competitor_offers o "
            f"JOIN competitor_scrapes s ON s.scrape_id = o.scrape_id "
            f"WHERE o.offer_id IN ({marks}) ORDER BY o.effective_price_cents, o.offer_id",
            ids,
        )
    )


# --- policies ---------------------------------------------------------------


def replace_policies(
    conn: sqlite3.Connection, policies: dict[str, Policy], version: int, now: datetime
) -> None:
    conn.execute("DELETE FROM pricing_policies")
    for price_class, p in policies.items():
        conn.execute(
            "INSERT INTO pricing_policies (price_class, strategy, undercut_cents, undercut_bp, "
            "target_margin_bp, max_decrease_bp, max_increase_bp, min_competitor_confidence, "
            "max_offer_age_hours, low_stock_threshold, round_to_cents, version, created_at, "
            "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                price_class,
                p.strategy.value,
                p.undercut_cents,
                p.undercut_bp,
                p.target_margin_bp,
                p.max_decrease_bp,
                p.max_increase_bp,
                p.min_competitor_confidence,
                p.max_offer_age_hours,
                p.low_stock_threshold,
                p.round_to_cents,
                version,
                iso(now),
                iso(now),
            ),
        )


# --- runs -------------------------------------------------------------------


def insert_run(
    conn: sqlite3.Connection, *, trigger: str, policy_version: int, now: datetime
) -> str:
    run_id = new_ulid()
    conn.execute(
        "INSERT INTO repricing_runs (run_id, started_at, trigger, policy_version, status) "
        "VALUES (?,?,?,?,'running')",
        (run_id, iso(now), trigger, policy_version),
    )
    return run_id


def finish_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    status: str,
    now: datetime,
    skus_considered: int | None = None,
    recommendations: int | None = None,
    offers_extracted: int | None = None,
    error: str | None = None,
) -> None:
    sets = ["finished_at = ?", "status = ?"]
    params: list[Any] = [iso(now), status]
    if skus_considered is not None:
        sets.append("skus_considered = ?")
        params.append(skus_considered)
    if recommendations is not None:
        sets.append("recommendations = ?")
        params.append(recommendations)
    if offers_extracted is not None:
        sets.append("offers_extracted = ?")
        params.append(offers_extracted)
    if error is not None:
        sets.append("error = ?")
        params.append(error)
    params.append(run_id)
    conn.execute(f"UPDATE repricing_runs SET {', '.join(sets)} WHERE run_id = ?", params)


def get_run(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM repricing_runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise NotFound(f"run {run_id}")
    return cast(sqlite3.Row, row)


# --- recommendations --------------------------------------------------------


def insert_recommendation(
    conn: sqlite3.Connection,
    rec: Recommendation,
    *,
    run_id: str,
    policy_version: int,
    now: datetime,
) -> str:
    rec_id = new_ulid()
    conn.execute(
        "INSERT INTO price_recommendations (recommendation_id, run_id, sku, current_price_cents, "
        "cost_price_cents, min_margin_bp, floor_price_cents, lowest_competitor_cents, "
        "competitor_offer_ids, raw_target_cents, proposed_price_cents, delta_cents, delta_bp, "
        "decision, reason_codes, explanation, policy_version, review_status, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            rec_id,
            run_id,
            rec.sku,
            rec.current_price_cents,
            rec.cost_price_cents,
            rec.min_margin_bp,
            rec.floor_price_cents,
            rec.lowest_competitor_cents,
            json.dumps(list(rec.competitor_offer_ids)),
            rec.raw_target_cents,
            rec.proposed_price_cents,
            rec.delta_cents,
            rec.delta_bp,
            rec.decision.value,
            json.dumps([c.value for c in rec.reason_codes]),
            rec.explanation,
            policy_version,
            ReviewStatus.PENDING.value,
            iso(now),
        ),
    )
    return rec_id


def get_recommendation(conn: sqlite3.Connection, rec_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM price_recommendations WHERE recommendation_id = ?", (rec_id,)
    ).fetchone()
    if row is None:
        raise NotFound(f"recommendation {rec_id}")
    return cast(sqlite3.Row, row)


def set_review_status(
    conn: sqlite3.Connection, rec_id: str, expected: ReviewStatus, new: ReviewStatus
) -> None:
    """Optimistic transition: fails if the row is no longer in ``expected``."""
    cur = conn.execute(
        "UPDATE price_recommendations SET review_status = ? "
        "WHERE recommendation_id = ? AND review_status = ?",
        (new.value, rec_id, expected.value),
    )
    if cur.rowcount != 1:
        raise ConcurrentUpdate(f"recommendation {rec_id} not in {expected.value}")


def expire_pending_for_sku(conn: sqlite3.Connection, sku: str) -> int:
    cur = conn.execute(
        "UPDATE price_recommendations SET review_status = 'expired' "
        "WHERE sku = ? AND review_status IN ('pending','approved')",
        (sku,),
    )
    return int(cur.rowcount)


def list_recommendations(
    conn: sqlite3.Connection,
    *,
    status: str | None = None,
    sku: str | None = None,
    run_id: str | None = None,
    decision: str | None = None,
    min_abs_delta_bp: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[sqlite3.Row], int]:
    where: list[str] = []
    params: list[Any] = []
    if status:
        where.append("review_status = ?")
        params.append(status)
    if sku:
        where.append("sku = ?")
        params.append(sku)
    if run_id:
        where.append("run_id = ?")
        params.append(run_id)
    if decision:
        where.append("decision = ?")
        params.append(decision)
    if min_abs_delta_bp is not None:
        where.append("ABS(delta_bp) >= ?")
        params.append(min_abs_delta_bp)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    total = conn.execute(
        f"SELECT COUNT(*) AS n FROM price_recommendations{clause}", params
    ).fetchone()["n"]
    rows = conn.execute(
        f"SELECT * FROM price_recommendations{clause} ORDER BY created_at DESC, "
        f"recommendation_id DESC LIMIT ? OFFSET ?",
        [*params, limit, offset],
    ).fetchall()
    return list(rows), int(total)


# --- review actions ---------------------------------------------------------


def insert_review_action(
    conn: sqlite3.Connection,
    *,
    recommendation_id: str,
    action: str,
    actor: str,
    override_price_cents: int | None,
    note: str | None,
    now: datetime,
) -> str:
    action_id = new_ulid()
    conn.execute(
        "INSERT INTO review_actions (action_id, recommendation_id, action, actor, "
        "override_price_cents, note, created_at) VALUES (?,?,?,?,?,?,?)",
        (action_id, recommendation_id, action, actor, override_price_cents, note, iso(now)),
    )
    return action_id


def actions_for_recommendation(conn: sqlite3.Connection, rec_id: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT * FROM review_actions WHERE recommendation_id = ? ORDER BY created_at",
            (rec_id,),
        )
    )


# --- price changes (append-only, hash chained) ------------------------------


def last_change_for_sku(conn: sqlite3.Connection, sku: str) -> sqlite3.Row | None:
    row = conn.execute(
        "SELECT * FROM price_changes WHERE sku = ? ORDER BY applied_at DESC, change_id DESC "
        "LIMIT 1",
        (sku,),
    ).fetchone()
    return cast("sqlite3.Row | None", row)


def insert_change(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO price_changes (change_id, sku, recommendation_id, action_id, "
        "old_price_cents, new_price_cents, source, actor, applied_at, prev_change_id, row_hash) "
        "VALUES (:change_id, :sku, :recommendation_id, :action_id, :old_price_cents, "
        ":new_price_cents, :source, :actor, :applied_at, :prev_change_id, :row_hash)",
        row,
    )


def changes_for_sku(conn: sqlite3.Connection, sku: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT * FROM price_changes WHERE sku = ? ORDER BY applied_at, change_id", (sku,)
        )
    )


def all_changes(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM price_changes ORDER BY sku, applied_at, change_id"))
