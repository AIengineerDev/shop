import numpy as np
import pandas as pd
import pytest

from shopfeed.config import Config
from shopfeed.data.synth import generate
from shopfeed.features.batch import compute_batch_state
from shopfeed.features.streaming import StreamingStore
from shopfeed.serving.pipeline import FeedService
from shopfeed.training.dataset import build_training_set, time_split
from shopfeed.training.train_ranker import fit_rankers
from shopfeed.training.train_retrieval import fit_retrieval

NOW = 1_750_000_000.0


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="session")
def world():
    return generate(seed=7, n_users=150, n_merchants=12, n_products=400, sessions_per_user=8)


@pytest.fixture(scope="session")
def streaming(world):
    s = StreamingStore()
    s.add_events(world["events"])
    return s


@pytest.fixture(scope="session")
def batch(world):
    return compute_batch_state(world["events"], world["impressions"], world["users"], world["products"], world["merchants"], NOW)


@pytest.fixture(scope="session")
def trained(world, streaming, cfg):
    ts = build_training_set(world, streaming, NOW, cfg.training.label_window_s)
    tr, va = time_split(ts.meta, cfg.training.val_fraction, cfg.training.label_window_s)
    retrieval = fit_retrieval(world, cfg, as_of=NOW, steps=120)
    light, heavy = fit_rankers(ts, tr, cfg, steps=200)
    return {"ts": ts, "train_idx": tr, "val_idx": va, "retrieval": retrieval, "light": light, "heavy": heavy}


def make_service(world, trained, cfg, **kw) -> FeedService:
    tables = {k: v.copy() for k, v in world.items()}
    return FeedService(tables, trained["retrieval"], trained["light"], trained["heavy"], cfg, now=NOW, **kw)


@pytest.fixture()
def service(world, trained, cfg):
    return make_service(world, trained, cfg)
