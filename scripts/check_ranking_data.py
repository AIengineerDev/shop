"""Run spec §9 data-readiness checks. Exit 1 if any check fails.

Usage: python scripts/check_ranking_data.py [--csv PATH] [--profile production|sample]
       [--no-sources]
"""

from __future__ import annotations

import argparse
import json
import sys

from shopfeed import config
from shopfeed.data.build import Sources
from shopfeed.data.checks import as_dicts, run_all
from shopfeed.schema import read_csv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.RANKING_CSV))
    ap.add_argument("--profile", choices=sorted(config.GATES), default="production")
    ap.add_argument("--no-sources", action="store_true", help="skip parquet-backed checks")
    args = ap.parse_args()

    df = read_csv(args.csv)
    src = None if args.no_sources else Sources.load()
    results = run_all(df, src, args.profile)
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.name:<24} {r.detail}")
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "readiness_report.json").write_text(
        json.dumps({"profile": args.profile, "checks": as_dicts(results)}, indent=2))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
