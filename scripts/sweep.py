"""M7: gain + hyperparameter sweep on validation, then ONE test read for the chosen config.

Usage: python scripts/sweep.py [--trials 30]
Writes reports/sweep.csv, reports/sweep.json and artifacts/ranker_v2/ for the chosen config.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from shopfeed import config
from shopfeed.eval.offline import write_html
from shopfeed.training.dataset import load_dataset
from shopfeed.training.sweep import gain_table, select, sweep
from shopfeed.training.train_lambdamart import run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--from-csv", action="store_true",
                    help="reuse reports/sweep.csv instead of re-running the search")
    args = ap.parse_args()

    ds = load_dataset()
    csv = config.REPORTS_DIR / "sweep.csv"
    if args.from_csv:
        results = pd.read_csv(csv)
    else:
        results = sweep(ds, n_trials=args.trials)
        config.REPORTS_DIR.mkdir(exist_ok=True)
        results.to_csv(csv, index=False)
    best = select(results)
    params = {"num_leaves": int(best["num_leaves"]), "learning_rate": float(best["learning_rate"]),
              "min_data_in_leaf": int(best["min_data_in_leaf"]),
              "feature_fraction": float(best["feature_fraction"]),
              "lambda_l2": float(best["lambda_l2"])}
    params["label_gain"] = gain_table(int(best["purchase_gain"]))

    out = config.ARTIFACTS_DIR / "ranker_v2"
    r = run(ds, out, params=params, position=True)  # single test read
    write_html(r["result"], config.REPORTS_DIR / "ranker_v2_test.html", "ranker_v2 test report")
    by_gain = results.groupby("purchase_gain")[["val_purchase_mrr", "val_ndcg@10"]].max()
    summary = {"chosen": {**params, "val_purchase_mrr": float(best["val_purchase_mrr"]),
                          "val_ndcg@10": float(best["val_ndcg@10"])},
               "best_val_by_gain": by_gain.reset_index().to_dict(orient="records"),
               "trials": len(results),
               "test": {"gates": r["metrics"]["gates"],
                        "deltas_vs_semantic": r["metrics"]["test"]["deltas_vs_semantic"],
                        "ipw_ndcg@10": r["metrics"]["ipw_ndcg@10"]}}
    (config.REPORTS_DIR / "sweep.json").write_text(json.dumps(summary, indent=2, default=str))
    print(by_gain.round(4))
    print(json.dumps(summary["chosen"], indent=1))
    o = r["result"]["overall"].pivot(index="metric", columns="scorer", values="value")
    print(o.loc[["ndcg@10", "purchase_mrr", "purchase@10"]].round(4))
    print(json.dumps(summary["test"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
