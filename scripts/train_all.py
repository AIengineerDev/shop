"""Train retrieval + light + heavy rankers on the logged impressions; save artifacts + metrics."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from shopfeed.config import Config
from shopfeed.data.store import load_tables
from shopfeed.eval.offline import per_head_metrics, ndcg_at_k
from shopfeed.features.streaming import StreamingStore
from shopfeed.ranking.value import value_score
from shopfeed.training.dataset import build_training_set, time_split
from shopfeed.training.train_ranker import fit_rankers, save_artifacts
from shopfeed.training.train_retrieval import fit_retrieval

cfg = Config()
tables = load_tables(cfg.data_dir)
now = json.loads((cfg.data_dir / "meta.json").read_text())["now"]
version = "ranker_v1"
t = time.time()
streaming = StreamingStore(); streaming.add_events(tables["events"])
ts = build_training_set(tables, streaming, now, cfg.training.label_window_s)
tr, va = time_split(ts.meta, cfg.training.val_fraction, cfg.training.label_window_s)
print(f"training set {ts.X.shape}, train {len(tr)} val {len(va)}, label prevalence {dict(zip(['click','dwell','save','atc','purchase','hide','return'], ts.Y.mean(0).round(3)))} ({time.time()-t:.1f}s)")
t = time.time()
retrieval = fit_retrieval(tables, cfg, as_of=now)
print(f"retrieval: loss {retrieval.history[0]:.3f} -> {retrieval.history[-1]:.3f} ({time.time()-t:.1f}s)")
t = time.time()
light, heavy = fit_rankers(ts, tr, cfg, version=version)
print(f"rankers: loss {heavy.history[0]:.3f} -> {heavy.history[-1]:.3f} ({time.time()-t:.1f}s)")
P = heavy.predict_proba(ts.X[va])
m = per_head_metrics(ts.Y[va], P)
prices = ts.meta.iloc[va][["product_id"]].merge(tables["products"][["product_id", "price"]], on="product_id", how="left").price.values
v = value_score(P, prices, cfg.value)
gain = ts.Y[va][:, 4] * 3 + ts.Y[va][:, 3] + ts.Y[va][:, 2] - 2 * ts.Y[va][:, 5]
ndcg = ndcg_at_k(v, np.clip(gain, 0, None), ts.meta.iloc[va].request_id.values, 20)
print("val AUC per head:", {k: round(x["auc"], 3) for k, x in m.items()})
print(f"val NDCG@20 (value score vs engagement gain): {ndcg:.3f}")
out = cfg.artifacts_dir / version
save_artifacts(out, retrieval, light, heavy)
(out / "metrics.json").write_text(json.dumps({"per_head": m, "ndcg20": ndcg, "n_train": int(len(tr)), "n_val": int(len(va))}, indent=2))
print("saved ->", out)
