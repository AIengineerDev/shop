"""Run the feed API: python scripts/serve.py  (then POST /feed)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import uvicorn
from shopfeed.config import Config
from shopfeed.data.store import load_tables
from shopfeed.serving.api import create_app
from shopfeed.serving.logger import ImpressionLogger
from shopfeed.serving.pipeline import FeedService
from shopfeed.training.train_ranker import load_artifacts

cfg = Config()
tables = load_tables(cfg.data_dir)
now = json.loads((cfg.data_dir / "meta.json").read_text())["now"]
retrieval, light, heavy = load_artifacts(cfg.artifacts_dir / "ranker_v1")
service = FeedService(tables, retrieval, light, heavy, cfg, now=now, logger=ImpressionLogger(cfg.root / "logs" / "impressions.jsonl"))
app = create_app(service)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
