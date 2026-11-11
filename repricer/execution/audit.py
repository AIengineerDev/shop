"""Audit queries: chain verification, SKU history, run reports."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from repricer.db import repo
from repricer.execution.hashing import row_hash


@dataclass(frozen=True)
class ChainBreak:
    sku: str
    change_id: str
    reason: str


def verify_chain(conn: sqlite3.Connection) -> list[ChainBreak]:
    breaks: list[ChainBreak] = []
    prev_by_sku: dict[str, str | None] = {}
    for row in repo.all_changes(conn):
        sku = row["sku"]
        expected_prev = prev_by_sku.get(sku)
        d = dict(row)
        if d["prev_change_id"] != expected_prev:
            breaks.append(
                ChainBreak(
                    sku,
                    d["change_id"],
                    f"prev_change_id {d['prev_change_id']!r} != expected {expected_prev!r}",
                )
            )
        if row_hash(d) != d["row_hash"]:
            breaks.append(ChainBreak(sku, d["change_id"], "row_hash mismatch (row tampered)"))
        prev_by_sku[sku] = d["change_id"]
    # Live price must equal the last change's new price for every SKU with history.
    for sku, last_id in prev_by_sku.items():
        if last_id is None:  # pragma: no cover - every SKU seen has at least one change
            continue
        last = conn.execute(
            "SELECT new_price_cents FROM price_changes WHERE change_id = ?", (last_id,)
        ).fetchone()
        inv = conn.execute(
            "SELECT current_price_cents FROM inventory WHERE sku = ?", (sku,)
        ).fetchone()
        if inv is not None and last is not None and inv[0] != last[0]:
            breaks.append(ChainBreak(sku, last_id, "inventory.current_price_cents != last change"))
    return breaks


def history(conn: sqlite3.Connection, sku: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in repo.changes_for_sku(conn, sku):
        d = dict(row)
        if d["recommendation_id"]:
            rec = repo.get_recommendation(conn, d["recommendation_id"])
            d["decision"] = rec["decision"]
            d["reason_codes"] = json.loads(rec["reason_codes"])
        if d["action_id"]:
            act = conn.execute(
                "SELECT action, note FROM review_actions WHERE action_id = ?", (d["action_id"],)
            ).fetchone()
            if act is not None:
                d["action"] = act["action"]
                d["note"] = act["note"]
        out.append(d)
    return out


def run_report(conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    run = dict(repo.get_run(conn, run_id))
    rows, _ = repo.list_recommendations(conn, run_id=run_id, limit=100_000)
    run["items"] = [{**dict(r), "reason_codes": json.loads(r["reason_codes"])} for r in rows]
    return run
