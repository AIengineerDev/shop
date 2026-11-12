"""Add AOV tier columns to data/users.parquet (serving snapshot; spec §2 user_aov_tier).

Adds aov_tier, aov_value, aov_order_count, aov_asof_ts as of meta.json "now".
Backs up the original to data/users.pre_aov.parquet. Refuses to run if the columns
already exist unless --force. Cut points go to data/aov_tiers.json.
Do NOT use these columns as training features: they are present-day (leakage).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys

import pandas as pd

from shopfeed import config
from shopfeed.data.aov import compute_aov

NEW_COLS = ["aov_tier", "aov_value", "aov_order_count", "aov_asof_ts"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    users_path = config.DATA_DIR / "users.parquet"
    users = pd.read_parquet(users_path)
    present = [c for c in NEW_COLS if c in users.columns]
    if present and not args.force:
        print(f"refusing: {present} already in users.parquet (use --force)", file=sys.stderr)
        return 1

    asof = float(json.loads((config.DATA_DIR / "meta.json").read_text())["now"])
    backup = config.DATA_DIR / "users.pre_aov.parquet"
    if not backup.exists():
        shutil.copy2(users_path, backup)

    events = pd.read_parquet(config.DATA_DIR / "events.parquet")
    aov, cuts = compute_aov(events, asof, pd.Index(users["user_id"]))
    users = users.drop(columns=present).merge(aov, on="user_id", how="left")
    users["aov_asof_ts"] = asof
    users.to_parquet(users_path, index=False)
    (config.DATA_DIR / "aov_tiers.json").write_text(json.dumps(
        {"asof_ts": asof, "cut_low_mid": cuts[0], "cut_mid_high": cuts[1],
         "lookback_s": config.AOV_LOOKBACK_S}, indent=2))
    print(f"users={len(users)} tiers={users['aov_tier'].value_counts().to_dict()} cuts={cuts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
