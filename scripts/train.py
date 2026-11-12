"""Train LambdaMART and write artifacts/ranker_v2/ (spec §7.4).

Usage: python scripts/train.py [--out DIR] [--no-position-debias] [--n-boot N] [--params JSON]
Prints the ship-gate table. Gates are reported, not enforced: promotion is a review decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shopfeed import config
from shopfeed.eval.offline import write_html
from shopfeed.training.dataset import load_dataset
from shopfeed.training.train_lambdamart import run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.ARTIFACTS_DIR / "ranker_v2"))
    ap.add_argument("--no-position-debias", action="store_true",
                    help="disable unbiased LambdaMART (on by default, spec §5.4)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--params", default=None, help="JSON dict overriding PARAMS")
    args = ap.parse_args()

    out = Path(args.out)
    r = run(load_dataset(), out, params=json.loads(args.params) if args.params else None,
            position=not args.no_position_debias, n_boot=args.n_boot)
    write_html(r["result"], config.REPORTS_DIR / f"{out.name}_test.html",
               f"{out.name} test report")
    o = r["result"]["overall"].pivot(index="metric", columns="scorer", values="value")
    print(o.loc[["ndcg@10", "purchase_mrr", "purchase@10", "mean_price@10"]].round(4))
    print("best_iteration:", r["metrics"]["best_iteration"])
    print(json.dumps(r["metrics"]["gates"], indent=1))
    print("leakage:", r["metrics"]["leakage_check"])
    print(r["importance"].head(8).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
