-- Repricer schema. Money = INTEGER cents. Percent = INTEGER basis points.
-- Append-only tables are protected by triggers; see bottom of file.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    applied_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pricing_policies (
    price_class               TEXT PRIMARY KEY,
    strategy                  TEXT NOT NULL CHECK (strategy IN ('match_lowest','undercut_lowest','hold','target_margin')),
    undercut_cents            INTEGER NOT NULL DEFAULT 0 CHECK (undercut_cents >= 0),
    undercut_bp               INTEGER NOT NULL DEFAULT 0 CHECK (undercut_bp >= 0),
    target_margin_bp          INTEGER NOT NULL DEFAULT 0 CHECK (target_margin_bp >= 0),
    max_decrease_bp           INTEGER NOT NULL CHECK (max_decrease_bp >= 0),
    max_increase_bp           INTEGER NOT NULL CHECK (max_increase_bp >= 0),
    min_competitor_confidence INTEGER NOT NULL CHECK (min_competitor_confidence BETWEEN 0 AND 100),
    max_offer_age_hours       INTEGER NOT NULL CHECK (max_offer_age_hours > 0),
    low_stock_threshold       INTEGER NOT NULL DEFAULT 0 CHECK (low_stock_threshold >= 0),
    round_to_cents            INTEGER NOT NULL DEFAULT 0 CHECK (round_to_cents BETWEEN 0 AND 99),
    version                   INTEGER NOT NULL,
    created_at                TEXT NOT NULL,
    updated_at                TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inventory (
    sku                  TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    cost_price_cents     INTEGER NOT NULL CHECK (cost_price_cents > 0),
    min_margin_bp        INTEGER NOT NULL CHECK (min_margin_bp >= 0),
    current_price_cents  INTEGER NOT NULL CHECK (current_price_cents > 0),
    stock_count          INTEGER NOT NULL CHECK (stock_count >= 0),
    price_class          TEXT NOT NULL DEFAULT 'default',
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS competitor_scrapes (
    scrape_id   INTEGER PRIMARY KEY,
    sku         TEXT,
    competitor  TEXT NOT NULL,
    raw_text    TEXT NOT NULL,
    scraped_at  TEXT NOT NULL,
    extracted   INTEGER NOT NULL DEFAULT 0 CHECK (extracted IN (0,1)),
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_scrapes_extracted ON competitor_scrapes(extracted);

CREATE TABLE IF NOT EXISTS competitor_offers (
    offer_id               INTEGER PRIMARY KEY,
    scrape_id              INTEGER NOT NULL REFERENCES competitor_scrapes(scrape_id),
    sku                    TEXT REFERENCES inventory(sku),
    competitor             TEXT NOT NULL,
    listed_price_cents     INTEGER CHECK (listed_price_cents IS NULL OR listed_price_cents >= 0),
    coupon_cents           INTEGER NOT NULL DEFAULT 0 CHECK (coupon_cents >= 0),
    coupon_bp              INTEGER NOT NULL DEFAULT 0 CHECK (coupon_bp >= 0),
    shipping_cents         INTEGER CHECK (shipping_cents IS NULL OR shipping_cents >= 0),
    effective_price_cents  INTEGER CHECK (effective_price_cents IS NULL OR effective_price_cents >= 0),
    currency               TEXT NOT NULL DEFAULT 'UNKNOWN',
    condition              TEXT NOT NULL DEFAULT 'unknown' CHECK (condition IN ('new','used','refurbished','unknown')),
    in_stock               INTEGER CHECK (in_stock IS NULL OR in_stock IN (0,1)),
    scraped_at             TEXT NOT NULL,
    extractor              TEXT NOT NULL,
    confidence             INTEGER NOT NULL CHECK (confidence BETWEEN 0 AND 100),
    status                 TEXT NOT NULL CHECK (status IN ('accepted','rejected','needs_review','superseded')),
    reject_reason          TEXT,
    reason_codes           TEXT NOT NULL DEFAULT '[]',
    raw_llm_json           TEXT,
    created_at             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_offers_sku_status ON competitor_offers(sku, status);
CREATE INDEX IF NOT EXISTS ix_offers_scrape ON competitor_offers(scrape_id);

CREATE TABLE IF NOT EXISTS repricing_runs (
    run_id           TEXT PRIMARY KEY,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    trigger          TEXT NOT NULL CHECK (trigger IN ('cli','api','schedule','test')),
    policy_version   INTEGER NOT NULL,
    skus_considered  INTEGER NOT NULL DEFAULT 0,
    recommendations  INTEGER NOT NULL DEFAULT 0,
    offers_extracted INTEGER NOT NULL DEFAULT 0,
    status           TEXT NOT NULL CHECK (status IN ('running','completed','failed')),
    error            TEXT
);

CREATE TABLE IF NOT EXISTS price_recommendations (
    recommendation_id        TEXT PRIMARY KEY,
    run_id                   TEXT NOT NULL REFERENCES repricing_runs(run_id),
    sku                      TEXT NOT NULL REFERENCES inventory(sku),
    current_price_cents      INTEGER NOT NULL,
    cost_price_cents         INTEGER NOT NULL,
    min_margin_bp            INTEGER NOT NULL,
    floor_price_cents        INTEGER NOT NULL,
    lowest_competitor_cents  INTEGER,
    competitor_offer_ids     TEXT NOT NULL DEFAULT '[]',
    raw_target_cents         INTEGER NOT NULL,
    proposed_price_cents     INTEGER NOT NULL,
    delta_cents              INTEGER NOT NULL,
    delta_bp                 INTEGER NOT NULL,
    decision                 TEXT NOT NULL CHECK (decision IN ('propose','hold','blocked')),
    reason_codes             TEXT NOT NULL,
    explanation              TEXT NOT NULL,
    policy_version           INTEGER NOT NULL,
    review_status            TEXT NOT NULL CHECK (review_status IN ('pending','approved','rejected','applied','expired')),
    created_at               TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_recs_sku_status ON price_recommendations(sku, review_status);
CREATE INDEX IF NOT EXISTS ix_recs_run ON price_recommendations(run_id);

CREATE TABLE IF NOT EXISTS review_actions (
    action_id             TEXT PRIMARY KEY,
    recommendation_id     TEXT NOT NULL REFERENCES price_recommendations(recommendation_id),
    action                TEXT NOT NULL CHECK (action IN ('approve','reject','override')),
    actor                 TEXT NOT NULL,
    override_price_cents  INTEGER,
    note                  TEXT,
    created_at            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS price_changes (
    change_id          TEXT PRIMARY KEY,
    sku                TEXT NOT NULL REFERENCES inventory(sku),
    recommendation_id  TEXT REFERENCES price_recommendations(recommendation_id),
    action_id          TEXT REFERENCES review_actions(action_id),
    old_price_cents    INTEGER NOT NULL,
    new_price_cents    INTEGER NOT NULL,
    source             TEXT NOT NULL CHECK (source IN ('review_approve','override','seed','migration')),
    actor              TEXT NOT NULL,
    applied_at         TEXT NOT NULL,
    prev_change_id     TEXT REFERENCES price_changes(change_id),
    row_hash           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_changes_sku ON price_changes(sku, applied_at);

-- ---------------------------------------------------------------------------
-- Append-only enforcement
-- ---------------------------------------------------------------------------

CREATE TRIGGER IF NOT EXISTS trg_price_changes_no_update
BEFORE UPDATE ON price_changes BEGIN
    SELECT RAISE(ABORT, 'price_changes is append-only');
END;
CREATE TRIGGER IF NOT EXISTS trg_price_changes_no_delete
BEFORE DELETE ON price_changes BEGIN
    SELECT RAISE(ABORT, 'price_changes is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_review_actions_no_update
BEFORE UPDATE ON review_actions BEGIN
    SELECT RAISE(ABORT, 'review_actions is append-only');
END;
CREATE TRIGGER IF NOT EXISTS trg_review_actions_no_delete
BEFORE DELETE ON review_actions BEGIN
    SELECT RAISE(ABORT, 'review_actions is append-only');
END;

-- price_recommendations: only review_status may change.
CREATE TRIGGER IF NOT EXISTS trg_recs_immutable
BEFORE UPDATE ON price_recommendations
WHEN NEW.recommendation_id   IS NOT OLD.recommendation_id
  OR NEW.run_id              IS NOT OLD.run_id
  OR NEW.sku                 IS NOT OLD.sku
  OR NEW.current_price_cents IS NOT OLD.current_price_cents
  OR NEW.cost_price_cents    IS NOT OLD.cost_price_cents
  OR NEW.min_margin_bp       IS NOT OLD.min_margin_bp
  OR NEW.floor_price_cents   IS NOT OLD.floor_price_cents
  OR NEW.lowest_competitor_cents IS NOT OLD.lowest_competitor_cents
  OR NEW.competitor_offer_ids IS NOT OLD.competitor_offer_ids
  OR NEW.raw_target_cents    IS NOT OLD.raw_target_cents
  OR NEW.proposed_price_cents IS NOT OLD.proposed_price_cents
  OR NEW.delta_cents         IS NOT OLD.delta_cents
  OR NEW.delta_bp            IS NOT OLD.delta_bp
  OR NEW.decision            IS NOT OLD.decision
  OR NEW.reason_codes        IS NOT OLD.reason_codes
  OR NEW.explanation         IS NOT OLD.explanation
  OR NEW.policy_version      IS NOT OLD.policy_version
  OR NEW.created_at          IS NOT OLD.created_at
BEGIN
    SELECT RAISE(ABORT, 'price_recommendations: only review_status is mutable');
END;
CREATE TRIGGER IF NOT EXISTS trg_recs_no_delete
BEFORE DELETE ON price_recommendations BEGIN
    SELECT RAISE(ABORT, 'price_recommendations is append-only');
END;

-- competitor_offers: only status may change (to 'superseded').
CREATE TRIGGER IF NOT EXISTS trg_offers_immutable
BEFORE UPDATE ON competitor_offers
WHEN NEW.offer_id IS NOT OLD.offer_id
  OR NEW.scrape_id IS NOT OLD.scrape_id
  OR NEW.sku IS NOT OLD.sku
  OR NEW.listed_price_cents IS NOT OLD.listed_price_cents
  OR NEW.coupon_cents IS NOT OLD.coupon_cents
  OR NEW.coupon_bp IS NOT OLD.coupon_bp
  OR NEW.shipping_cents IS NOT OLD.shipping_cents
  OR NEW.effective_price_cents IS NOT OLD.effective_price_cents
  OR NEW.currency IS NOT OLD.currency
  OR NEW.condition IS NOT OLD.condition
  OR NEW.confidence IS NOT OLD.confidence
  OR NEW.raw_llm_json IS NOT OLD.raw_llm_json
  OR NEW.created_at IS NOT OLD.created_at
BEGIN
    SELECT RAISE(ABORT, 'competitor_offers: only status is mutable');
END;
CREATE TRIGGER IF NOT EXISTS trg_offers_no_delete
BEFORE DELETE ON competitor_offers BEGIN
    SELECT RAISE(ABORT, 'competitor_offers is append-only');
END;
