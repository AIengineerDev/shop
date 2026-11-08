"""Offline evaluation + simulated guardrails + simulated A/B for the trained model."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from shopfeed.config import Config, ValueWeights
from shopfeed.data.store import load_tables
from shopfeed.eval.ab import assign_bucket, compare
from shopfeed.eval.guardrails import guardrail_report, failed
from shopfeed.eval.offline import calibration_table
from shopfeed.eval.simulator import simulate_responses, responses_to_events
from shopfeed.schema import FeedRequest
from shopfeed.serving.pipeline import FeedService
from shopfeed.training.train_ranker import load_artifacts

cfg = Config()
tables = load_tables(cfg.data_dir)
now = json.loads((cfg.data_dir / "meta.json").read_text())["now"]
art_dir = cfg.artifacts_dir / "ranker_v1"
retrieval, light, heavy = load_artifacts(art_dir)
metrics = json.loads((art_dir / "metrics.json").read_text())
print("== offline (validation split) ==")
for h, m in metrics["per_head"].items():
    print(f"  {h:9s} auc={m['auc']:.3f} logloss={m['logloss']:.3f} prevalence={m['prevalence']:.3f}")
print(f"  ndcg@20={metrics['ndcg20']:.3f}")

def serve_and_simulate(weights: ValueWeights, users: pd.DataFrame, seed: int, policy: str = "model"):
    c = Config(); c.value = weights
    svc = FeedService(tables, retrieval, light, heavy, c, now=now, seed=seed, policy=policy)
    rng = np.random.default_rng(seed)
    pages, events, lat = [], [], []
    for u in users.itertuples(index=False):
        resp = svc.recommend(FeedRequest(user_id=u.user_id, country=u.country, session_id="eval", now=now))
        lat.append(resp.timings_ms["total"])
        if not resp.cards:
            continue
        page = pd.DataFrame([c.model_dump() for c in resp.cards])
        sim = simulate_responses(page, users.set_index("user_id").loc[u.user_id], tables["products"], tables["merchants"], rng)
        pages.append(sim.assign(user_id=u.user_id, ts=now))
        events.append(responses_to_events(sim, u.user_id, now, start_id=len(events) * 1000))
    pages = pd.concat(pages, ignore_index=True)
    events = pd.concat(events, ignore_index=True)
    pages = pages.merge(tables["products"][["product_id", "published_at"]], on="product_id", how="left")
    return pages, events, np.array(lat)

rng = np.random.default_rng(0)
sample = tables["users"].sample(min(300, len(tables["users"])), random_state=0)
t = time.time()
pages, events, lat = serve_and_simulate(cfg.value, sample, seed=0)
print(f"\n== simulated serving: {len(sample)} users, {len(pages)} cards, {time.time()-t:.1f}s ==")
print(f"  latency ms: p50={np.percentile(lat,50):.1f} p95={np.percentile(lat,95):.1f} budget={cfg.latency_budget_ms}")
print(f"  simulated rates: click={pages.click.mean():.3f} purchase={pages.purchase.mean():.3f} hide={pages.hide.mean():.3f} gmv/user={pages.order_value.sum()/len(sample):.2f}")
print(f"  source mix: {pages.source.value_counts(normalize=True).round(2).to_dict()}")
rep = guardrail_report(pages, events, cfg.guardrails, cfg.rerank.fresh_window_s)
print("  guardrails:", {k: f"{v.value:.3f} ({'ok' if v.passed else 'FAIL'} vs {v.direction} {v.threshold})" for k, v in rep.items()})
print("  failed:", failed(rep) or "none")
cal = calibration_table(pages.p_purchase.values if "p_purchase" in pages else np.zeros(len(pages)), pages.purchase.values.astype(float)) if "p_purchase" in pages else None

print("\n== simulated A/B: control = random policy over eligible candidates, treatment = model ==")
users = sample.copy()
users["bucket"] = [assign_bucket(u, "model_vs_random") for u in users.user_id]
pc, ec, _ = serve_and_simulate(cfg.value, users[users.bucket == "control"], seed=1, policy="random")
pt, et, _ = serve_and_simulate(cfg.value, users[users.bucket == "treatment"], seed=1, policy="model")
per_user = pd.concat([
    pc.groupby("user_id").order_value.sum().rename("metric").reset_index().assign(bucket="control"),
    pt.groupby("user_id").order_value.sum().rename("metric").reset_index().assign(bucket="treatment"),
])
print(" ", compare(per_user, n_boot=500))
for name, (pg, ev) in {"control": (pc, ec), "treatment": (pt, et)}.items():
    r = guardrail_report(pg, ev, cfg.guardrails)
    print(f"  {name} guardrails failed: {failed(r) or 'none'}")
