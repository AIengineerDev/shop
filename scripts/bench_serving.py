"""M8: end-to-end Ranker.rank() latency on real test-split requests (spec §12.4).

Usage: python scripts/bench_serving.py [--requests 300] [--candidates 20|200]
With --candidates 200, ten test groups are concatenated into one 200-candidate request.
Writes reports/serving_bench.json.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from shopfeed import config
from shopfeed.serving.ranker import Ranker
from shopfeed.training.dataset import load_dataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--candidates", type=int, default=20)
    args = ap.parse_args()

    ds = load_dataset()
    te, _ = ds.part("test")
    ranker = Ranker.load(config.ARTIFACTS_DIR / "ranker_v2", ds.store, timeout_s=10.0)
    qids = te["qid"].unique()
    per = max(1, args.candidates // 20)
    reqs = []
    for i in range(args.requests):
        chunk = qids[(i * per) % len(qids): (i * per) % len(qids) + per]
        r = te[te["qid"].isin(chunk)].copy()
        r["qid"] = int(chunk[0])
        reqs.append(r.reset_index(drop=True))
    for r in reqs[:10]:
        ranker.rank(r)
    timings = [ranker.rank(r).timings_ms for r in reqs]
    t = pd.DataFrame(timings)
    out = {"candidates_per_request": int(len(reqs[0])), "requests": len(reqs),
           "fallbacks": ranker.counters["fallback"],
           **{f"{c}_p50_ms": float(np.percentile(t[c], 50)) for c in t},
           **{f"{c}_p99_ms": float(np.percentile(t[c], 99)) for c in t}}
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / f"serving_bench_{args.candidates}.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
