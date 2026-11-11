-- 0001: convert the "given" REAL-money inventory/scrapes tables to integer cents / bp.
-- Executed by repricer.db.migrate after the precision assertion passes.
-- Preconditions: inventory_legacy and competitor_scrapes_legacy exist; new tables are empty.

INSERT INTO inventory (sku, title, cost_price_cents, min_margin_bp, current_price_cents,
                       stock_count, price_class, created_at, updated_at)
SELECT
    sku,
    title,
    CAST(ROUND(cost_price * 100) AS INTEGER),
    CAST(ROUND(min_margin_percent * 100) AS INTEGER),
    CAST(ROUND(current_price * 100) AS INTEGER),
    stock_count,
    COALESCE(price_class, 'default'),
    :now,
    :now
FROM inventory_legacy;

INSERT INTO competitor_scrapes (scrape_id, sku, competitor, raw_text, scraped_at, extracted, created_at)
SELECT
    scrape_id,
    sku,
    COALESCE(competitor, 'unknown'),
    raw_text,
    COALESCE(scraped_at, :now),
    0,
    :now
FROM competitor_scrapes_legacy;

DROP TABLE inventory_legacy;
DROP TABLE competitor_scrapes_legacy;
