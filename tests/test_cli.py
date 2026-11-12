import sqlite3
from pathlib import Path

from catalog_onboard.cli import main


def test_ingest_cli_summary_and_idempotent(db_path: Path, capsys):
    assert main(["ingest", "--db", str(db_path), "--llm", "fake", "--limit", "10"]) == 0
    out = capsys.readouterr().out
    assert "processed" in out and "draft" in out and "quarantined" in out and "errors" in out
    assert main(["ingest", "--db", str(db_path), "--llm", "fake", "--limit", "10"]) == 0
    out2 = capsys.readouterr().out
    assert "processed                    0" in out2
    conn = sqlite3.connect(str(db_path))
    n = (
        conn.execute("SELECT COUNT(*) FROM draft_catalog").fetchone()[0]
        + conn.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0]
    )
    assert n == 10


def test_cli_dry_run(db_path: Path, capsys):
    assert main(["ingest", "--db", str(db_path), "--llm", "fake", "--limit", "3", "--dry-run"]) == 0
    conn = sqlite3.connect(str(db_path))
    assert conn.execute("SELECT COUNT(*) FROM processed_items").fetchone()[0] == 0


def test_cli_config_errors(tmp_path: Path, capsys):
    assert main(["ingest", "--db", str(tmp_path / "missing.sqlite")]) == 2
    empty = tmp_path / "empty.sqlite"
    sqlite3.connect(str(empty)).close()
    assert main(["ingest", "--db", str(empty)]) == 2
    assert "canonical_categories" in capsys.readouterr().err


def test_cli_supplier_filter(db_path: Path, capsys):
    assert main(["ingest", "--db", str(db_path), "--llm", "fake", "--supplier", "sup_acme"]) == 0
    conn = sqlite3.connect(str(db_path))
    sup = conn.execute(
        "SELECT DISTINCT supplier_id FROM draft_catalog UNION SELECT DISTINCT supplier_id FROM quarantine"
    ).fetchall()
    assert sup == [("sup_acme",)]


def test_non_utf8_row_is_quarantined_not_fatal(db_path: Path, capsys):
    """AC-1.5 / AC-3.5 / AC-3.6: one row with invalid UTF-8 bytes must not abort the batch."""
    conn = sqlite3.connect(str(db_path))
    fid = conn.execute("SELECT feed_item_id FROM incoming_supplier_feeds ORDER BY received_at LIMIT 1 OFFSET 3"
                       ).fetchone()[0]
    conn.execute("UPDATE incoming_supplier_feeds SET raw_title = CAST(x'4d656e2773fffe2053686f727473' AS TEXT) "
                 "WHERE feed_item_id = ?", (fid,))
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM incoming_supplier_feeds").fetchone()[0]
    conn.close()

    assert main(["ingest", "--db", str(db_path), "--llm", "fake"]) == 0
    conn = sqlite3.connect(str(db_path))
    done = (conn.execute("SELECT COUNT(*) FROM draft_catalog").fetchone()[0]
            + conn.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0])
    assert done == total
    codes = conn.execute("SELECT reason_codes_json FROM quarantine WHERE feed_item_id = ?", (fid,)).fetchone()[0]
    assert "TEXT_ENCODING_INVALID:raw_title" in codes and "PIPELINE_ERROR" not in codes
