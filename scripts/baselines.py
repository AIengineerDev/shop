"""M4: score §6.1 baselines on the test split; write reports/baselines.json and .html."""

from __future__ import annotations

import json
import sys

import pandas as pd

from shopfeed import config
from shopfeed.eval.offline import evaluate, write_html
from shopfeed.training.baselines import SKIPPED, all_scores, train_pointwise
from shopfeed.training.dataset import load_dataset


def main() -> int:
    ds = load_dataset()
    tr_rows, tr_X = ds.part("train")
    va_rows, va_X = ds.part("val")
    te_rows, te_X = ds.part("test")
    booster = train_pointwise(tr_X, tr_rows["clicked"], va_X, va_rows["clicked"])
    scores = all_scores(te_rows, te_X, booster)
    df = pd.concat([te_rows, scores], axis=1)
    res = evaluate(df, list(scores.columns), reference="logged")

    out = {
        "split": ds.split.to_dict(),
        "skipped": SKIPPED,
        "pointwise_best_iteration": booster.best_iteration,
        "overall": res["overall"].to_dict(orient="records"),
        "deltas_vs_logged": res["deltas"].to_dict(orient="records"),
    }
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "baselines.json").write_text(json.dumps(out, indent=2))
    write_html(res, config.REPORTS_DIR / "baselines.html", "Baselines (test split)")
    print(res["overall"].pivot(index="metric", columns="scorer", values="value").round(4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
