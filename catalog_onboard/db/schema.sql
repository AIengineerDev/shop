-- Output tables created by the ingestion engine. Starter tables
-- (canonical_categories, incoming_supplier_feeds) are provided by the caller.

CREATE TABLE IF NOT EXISTS draft_catalog (
    listing_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    feed_item_id     TEXT NOT NULL UNIQUE,
    supplier_id      TEXT NOT NULL,
    category_id      TEXT NOT NULL,
    clean_title      TEXT NOT NULL,
    attributes_json  TEXT NOT NULL,
    reason_codes_json TEXT NOT NULL DEFAULT '[]',
    price            TEXT,
    fingerprint      TEXT NOT NULL,
    pipeline_run_id  TEXT NOT NULL,
    rules_version    TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_draft_supplier ON draft_catalog(supplier_id);
CREATE INDEX IF NOT EXISTS idx_draft_category ON draft_catalog(category_id);
CREATE INDEX IF NOT EXISTS idx_draft_created ON draft_catalog(created_at, listing_id);
CREATE INDEX IF NOT EXISTS idx_draft_fingerprint ON draft_catalog(fingerprint);

CREATE TABLE IF NOT EXISTS quarantine (
    quarantine_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    feed_item_id         TEXT NOT NULL UNIQUE,
    supplier_id          TEXT NOT NULL,
    proposed_category_id TEXT,
    candidate_json       TEXT,
    reason_codes_json    TEXT NOT NULL,
    pipeline_run_id      TEXT NOT NULL,
    rules_version        TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    resolved_at          TEXT,
    resolution           TEXT
);
CREATE INDEX IF NOT EXISTS idx_quar_supplier ON quarantine(supplier_id);
CREATE INDEX IF NOT EXISTS idx_quar_category ON quarantine(proposed_category_id);
CREATE INDEX IF NOT EXISTS idx_quar_created ON quarantine(created_at, quarantine_id);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    feed_item_id  TEXT NOT NULL,
    stage         TEXT NOT NULL,
    seq           INTEGER NOT NULL,
    input_json    TEXT,
    output_json   TEXT,
    llm_raw_json  TEXT,
    duration_ms   INTEGER NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_item ON pipeline_runs(feed_item_id, id);
CREATE INDEX IF NOT EXISTS idx_runs_run ON pipeline_runs(run_id);

-- One row per (feed_item_id, rules_version) processed; drives idempotency.
CREATE TABLE IF NOT EXISTS processed_items (
    feed_item_id   TEXT NOT NULL,
    rules_version  TEXT NOT NULL,
    status         TEXT NOT NULL,
    run_id         TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (feed_item_id, rules_version)
);
