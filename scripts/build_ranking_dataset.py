"""Regenerate data/ranking_items.csv from parquet: build -> §9.1 clean -> schema validate.

Usage: python scripts/build_ranking_dataset.py [--out PATH]
Writes reports/build_report.json and reports/clean_report.json. Exits 1 if cleaning fails.
"""

from __future__ import annotations

import argparse
import io
import json
import sys

from shopfeed import config
from shopfeed.data.build import BuildReport, Sources, build_rows
from shopfeed.data.clean import CleanError, clean, read_raw, write_report
from shopfeed.schema import validate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.RANKING_CSV))
    args = ap.parse_args()

    report = BuildReport()
    built = build_rows(Sources.load(), report)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "build_report.json").write_text(json.dumps(report.counts, indent=2))

    buf = io.StringIO()
    built.to_csv(buf, index=False)
    buf.seek(0)
    try:
        df, clean_report = clean(read_raw(buf))
    except CleanError as e:
        write_report(e.report, config.REPORTS_DIR / "clean_report.json")
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    write_report(clean_report, config.REPORTS_DIR / "clean_report.json")
    validate(df)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: rows={len(df)} groups={df['qid'].nunique()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
