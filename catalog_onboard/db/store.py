"""SQLite access layer. Thin, explicit SQL; no ORM."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from catalog_onboard.models import (
    AttrValue,
    CandidateListing,
    DraftListing,
    PipelineRunRecord,
    QuarantineItem,
    RawFeedRow,
    ReasonCode,
)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
FEED_KNOWN_COLUMNS = {"feed_item_id", "supplier_id", "raw_title", "raw_bullets", "raw_tags", "raw_price", "received_at"}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _json_default(o: Any) -> Any:
    if isinstance(o, Decimal):
        return str(o)
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, sort_keys=True, ensure_ascii=False)


def _decode_text(raw: bytes) -> str:
    """Never raise on bad bytes (AC-1.5): undecodable bytes become U+FFFD, which the clean
    stage turns into TEXT_ENCODING_INVALID:<field>. Valid UTF-8 is unchanged."""
    return raw.decode("utf-8", errors="replace")


def open_readwrite(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.text_factory = _decode_text
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def open_readonly(path: str | Path) -> sqlite3.Connection:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    conn.text_factory = _decode_text
    conn.row_factory = sqlite3.Row
    return conn


class Store:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ----- schema -----------------------------------------------------------

    def migrate(self) -> None:
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(draft_catalog)")}
        if "candidate_json" not in cols:  # DBs created before v1.1
            self.conn.execute("ALTER TABLE draft_catalog ADD COLUMN candidate_json TEXT")
        self.conn.commit()

    def iter_stored_candidates(self, *, limit: int | None = None) -> Iterator[tuple[str, str]]:
        """(feed_item_id, candidate_json) for every current outcome that kept its candidate."""
        sql = (
            "SELECT feed_item_id, candidate_json FROM draft_catalog WHERE candidate_json IS NOT NULL "
            "UNION ALL SELECT feed_item_id, candidate_json FROM quarantine WHERE candidate_json IS NOT NULL "
            "ORDER BY feed_item_id"
        )
        params: list[Any] = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        for row in self.conn.execute(sql, params).fetchall():
            yield str(row[0]), str(row[1])

    def has_starter_tables(self) -> bool:
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('canonical_categories','incoming_supplier_feeds')"
        ).fetchall()
        return len(rows) == 2

    # ----- starter reads ----------------------------------------------------

    def iter_categories(self) -> Iterator[sqlite3.Row]:
        yield from self.conn.execute(
            "SELECT category_id, name, parent_id, required_attributes_json, prohibited_keywords "
            "FROM canonical_categories ORDER BY category_id"
        )

    def iter_feed_rows(self, *, supplier_id: str | None = None, limit: int | None = None) -> Iterator[RawFeedRow]:
        sql = "SELECT * FROM incoming_supplier_feeds"
        params: list[Any] = []
        if supplier_id:
            sql += " WHERE supplier_id = ?"
            params.append(supplier_id)
        sql += " ORDER BY received_at, feed_item_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        for row in self.conn.execute(sql, params):
            d = dict(row)
            extra = {k: v for k, v in d.items() if k not in FEED_KNOWN_COLUMNS}
            yield RawFeedRow(
                feed_item_id=str(d["feed_item_id"]),
                supplier_id=str(d["supplier_id"]),
                raw_title=d.get("raw_title"),
                raw_bullets=d.get("raw_bullets"),
                raw_tags=d.get("raw_tags"),
                raw_price=None if d.get("raw_price") is None else str(d["raw_price"]),
                received_at=d.get("received_at"),
                extra=extra,
            )

    def count_feed_rows(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM incoming_supplier_feeds").fetchone()[0])

    # ----- idempotency ------------------------------------------------------

    def processed_status(self, feed_item_id: str, rules_version: str) -> str | None:
        row = self.conn.execute(
            "SELECT status FROM processed_items WHERE feed_item_id=? AND rules_version=?",
            (feed_item_id, rules_version),
        ).fetchone()
        return None if row is None else str(row["status"])

    def mark_processed(self, feed_item_id: str, rules_version: str, status: str, run_id: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO processed_items(feed_item_id, rules_version, status, run_id, created_at) "
            "VALUES (?,?,?,?,?)",
            (feed_item_id, rules_version, status, run_id, _now()),
        )

    def supersede(self, feed_item_id: str) -> None:
        """Remove any prior draft/quarantine outcome for this item (pipeline_runs are kept)."""
        self.conn.execute("DELETE FROM draft_catalog WHERE feed_item_id=?", (feed_item_id,))
        self.conn.execute("DELETE FROM quarantine WHERE feed_item_id=?", (feed_item_id,))

    # ----- writes -----------------------------------------------------------

    def insert_draft(
        self,
        *,
        feed_item_id: str,
        supplier_id: str,
        category_id: str,
        clean_title: str,
        attributes: dict[str, AttrValue],
        reason_codes: Sequence[ReasonCode],
        price: Decimal | None,
        fingerprint: str,
        run_id: str,
        rules_version: str,
        candidate: dict[str, Any] | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO draft_catalog(feed_item_id, supplier_id, category_id, clean_title, attributes_json, "
            "reason_codes_json, price, fingerprint, pipeline_run_id, rules_version, created_at, candidate_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                feed_item_id,
                supplier_id,
                category_id,
                clean_title,
                dumps(attributes),
                dumps([r.model_dump() for r in reason_codes]),
                None if price is None else str(price),
                fingerprint,
                run_id,
                rules_version,
                _now(),
                None if candidate is None else dumps(candidate),
            ),
        )
        return int(cur.lastrowid or 0)

    def insert_quarantine(
        self,
        *,
        feed_item_id: str,
        supplier_id: str,
        proposed_category_id: str | None,
        candidate: dict[str, Any] | None,
        reason_codes: Sequence[ReasonCode],
        run_id: str,
        rules_version: str,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO quarantine(feed_item_id, supplier_id, proposed_category_id, candidate_json, "
            "reason_codes_json, pipeline_run_id, rules_version, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                feed_item_id,
                supplier_id,
                proposed_category_id,
                None if candidate is None else dumps(candidate),
                dumps([r.model_dump() for r in reason_codes]),
                run_id,
                rules_version,
                _now(),
            ),
        )
        return int(cur.lastrowid or 0)

    def insert_run(
        self,
        *,
        run_id: str,
        feed_item_id: str,
        stage: str,
        seq: int,
        input_obj: Any,
        output_obj: Any,
        llm_raw: Any,
        duration_ms: int,
    ) -> None:
        self.conn.execute(
            "INSERT INTO pipeline_runs(run_id, feed_item_id, stage, seq, input_json, output_json, llm_raw_json, "
            "duration_ms, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                feed_item_id,
                stage,
                seq,
                None if input_obj is None else dumps(input_obj),
                None if output_obj is None else dumps(output_obj),
                None if llm_raw is None else dumps(llm_raw),
                duration_ms,
                _now(),
            ),
        )

    def fingerprint_exists(self, fingerprint: str, *, exclude_feed_item_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM draft_catalog WHERE fingerprint=? AND feed_item_id<>? LIMIT 1",
            (fingerprint, exclude_feed_item_id),
        ).fetchone()
        return row is not None

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

    # ----- reads for API ----------------------------------------------------

    @staticmethod
    def _draft_from_row(row: sqlite3.Row) -> DraftListing:
        return DraftListing(
            listing_id=row["listing_id"],
            feed_item_id=row["feed_item_id"],
            supplier_id=row["supplier_id"],
            category_id=row["category_id"],
            clean_title=row["clean_title"],
            attributes=json.loads(row["attributes_json"]),
            price=None if row["price"] is None else Decimal(row["price"]),
            pipeline_run_id=row["pipeline_run_id"],
            rules_version=row["rules_version"],
            created_at=datetime.fromisoformat(row["created_at"]),
            reason_codes=[ReasonCode(**r) for r in json.loads(row["reason_codes_json"])],
        )

    @staticmethod
    def _quarantine_from_row(row: sqlite3.Row) -> QuarantineItem:
        return QuarantineItem(
            quarantine_id=row["quarantine_id"],
            feed_item_id=row["feed_item_id"],
            supplier_id=row["supplier_id"],
            proposed_category_id=row["proposed_category_id"],
            candidate=None
            if row["candidate_json"] is None
            else CandidateListing.model_validate(json.loads(row["candidate_json"])),
            reason_codes=[ReasonCode(**r) for r in json.loads(row["reason_codes_json"])],
            pipeline_run_id=row["pipeline_run_id"],
            rules_version=row["rules_version"],
            created_at=datetime.fromisoformat(row["created_at"]),
            resolved_at=None if row["resolved_at"] is None else datetime.fromisoformat(row["resolved_at"]),
            resolution=row["resolution"],
        )

    def list_drafts(
        self,
        *,
        supplier_id: str | None,
        category_id: str | None,
        q: str | None,
        after: tuple[str, int] | None,
        limit: int,
    ) -> list[DraftListing]:
        where: list[str] = []
        params: list[Any] = []
        if supplier_id:
            where.append("supplier_id = ?")
            params.append(supplier_id)
        if category_id:
            where.append("category_id = ?")
            params.append(category_id)
        if q:
            where.append("lower(clean_title) LIKE ?")
            params.append(f"%{q.lower()}%")
        if after:
            where.append("(created_at, listing_id) > (?, ?)")
            params.extend(after)
        sql = "SELECT * FROM draft_catalog"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at, listing_id LIMIT ?"
        params.append(limit)
        return [self._draft_from_row(r) for r in self.conn.execute(sql, params)]

    def get_draft(self, listing_id: int) -> DraftListing | None:
        row = self.conn.execute("SELECT * FROM draft_catalog WHERE listing_id=?", (listing_id,)).fetchone()
        return None if row is None else self._draft_from_row(row)

    def list_quarantine(
        self,
        *,
        supplier_id: str | None,
        reason_code: str | None,
        proposed_category_id: str | None,
        after: tuple[str, int] | None,
        limit: int,
    ) -> list[QuarantineItem]:
        where: list[str] = []
        params: list[Any] = []
        if supplier_id:
            where.append("supplier_id = ?")
            params.append(supplier_id)
        if proposed_category_id:
            where.append("proposed_category_id = ?")
            params.append(proposed_category_id)
        if reason_code:
            # match code exactly or as prefix family, e.g. ATTR_MISSING matches ATTR_MISSING:size
            where.append(
                "EXISTS (SELECT 1 FROM json_each(quarantine.reason_codes_json) je "
                "WHERE json_extract(je.value,'$.code') = ? OR json_extract(je.value,'$.code') LIKE ?)"
            )
            params.extend([reason_code, f"{reason_code}:%"])
        if after:
            where.append("(created_at, quarantine_id) > (?, ?)")
            params.extend(after)
        sql = "SELECT * FROM quarantine"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at, quarantine_id LIMIT ?"
        params.append(limit)
        return [self._quarantine_from_row(r) for r in self.conn.execute(sql, params)]

    def get_quarantine(self, quarantine_id: int) -> QuarantineItem | None:
        row = self.conn.execute("SELECT * FROM quarantine WHERE quarantine_id=?", (quarantine_id,)).fetchone()
        return None if row is None else self._quarantine_from_row(row)

    def trace(self, feed_item_id: str) -> list[PipelineRunRecord]:
        rows = self.conn.execute(
            "SELECT * FROM pipeline_runs WHERE feed_item_id=? ORDER BY id", (feed_item_id,)
        ).fetchall()
        return [
            PipelineRunRecord(
                run_id=r["run_id"],
                feed_item_id=r["feed_item_id"],
                stage=r["stage"],
                seq=r["seq"],
                input=None if r["input_json"] is None else json.loads(r["input_json"]),
                output=None if r["output_json"] is None else json.loads(r["output_json"]),
                llm_raw=None if r["llm_raw_json"] is None else json.loads(r["llm_raw_json"]),
                duration_ms=r["duration_ms"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    def stats(self) -> dict[str, Any]:
        drafts = int(self.conn.execute("SELECT COUNT(*) FROM draft_catalog").fetchone()[0])
        quarantined = int(self.conn.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0])
        top_codes = [
            {"code": r[0], "count": int(r[1])}
            for r in self.conn.execute(
                "SELECT json_extract(je.value,'$.code') AS code, COUNT(*) FROM quarantine, "
                "json_each(quarantine.reason_codes_json) je GROUP BY code ORDER BY COUNT(*) DESC, code LIMIT 20"
            )
        ]
        per_supplier = [
            {
                "supplier_id": r[0],
                "draft": int(r[1]),
                "quarantined": int(r[2]),
                "quarantine_rate": (int(r[2]) / (int(r[1]) + int(r[2]))) if (int(r[1]) + int(r[2])) else 0.0,
            }
            for r in self.conn.execute(
                "SELECT s, SUM(d), SUM(q) FROM ("
                "  SELECT supplier_id AS s, 1 AS d, 0 AS q FROM draft_catalog"
                "  UNION ALL SELECT supplier_id, 0, 1 FROM quarantine"
                ") GROUP BY s ORDER BY s"
            )
        ]
        return {
            "draft": drafts,
            "quarantined": quarantined,
            "top_reason_codes": top_codes,
            "per_supplier": per_supplier,
        }
