import sqlite3
from pathlib import Path


def test_build_is_deterministic(tmp_path: Path, make_starter_db):
    a, b = tmp_path / "a.sqlite", tmp_path / "b.sqlite"
    make_starter_db.build(a, seed=3, n_rows=60)
    make_starter_db.build(b, seed=3, n_rows=60)

    def rows(p: Path) -> list[tuple]:
        return sqlite3.connect(str(p)).execute("SELECT * FROM incoming_supplier_feeds ORDER BY feed_item_id").fetchall()

    assert rows(a) == rows(b)


def test_counts_and_sample_row(starter_db_path: Path, make_starter_db):
    conn = sqlite3.connect(str(starter_db_path))
    assert conn.execute("SELECT COUNT(*) FROM incoming_supplier_feeds").fetchone()[0] == 120
    assert conn.execute("SELECT COUNT(*) FROM canonical_categories").fetchone()[0] == len(make_starter_db.CATEGORIES)
    title = conn.execute("SELECT raw_title FROM incoming_supplier_feeds WHERE feed_item_id='SAMPLE-0001'").fetchone()[0]
    assert title == make_starter_db.SAMPLE_TITLE
    defects = conn.execute("SELECT COUNT(*) FROM incoming_supplier_feeds WHERE injected_defect IS NOT NULL").fetchone()[
        0
    ]
    assert defects == 12
