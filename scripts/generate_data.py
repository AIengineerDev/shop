"""Generate the synthetic Shop world. Usage: python scripts/generate_data.py --seed 42 --users 2000 --products 4000"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shopfeed.config import Config
from shopfeed.data.store import save_tables
from shopfeed.data.synth import generate

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=42)
p.add_argument("--users", type=int, default=2000)
p.add_argument("--merchants", type=int, default=60)
p.add_argument("--products", type=int, default=4000)
p.add_argument("--sessions", type=float, default=6.0)
a = p.parse_args()
cfg = Config()
t = time.time()
tables = generate(seed=a.seed, n_users=a.users, n_merchants=a.merchants, n_products=a.products, sessions_per_user=a.sessions)
save_tables(tables, cfg.data_dir)
now = 1_750_000_000.0
(cfg.data_dir / "meta.json").write_text(json.dumps({"now": now, "seed": a.seed}))
print({k: len(v) for k, v in tables.items()}, f"{time.time()-t:.1f}s -> {cfg.data_dir}")
