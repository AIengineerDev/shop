import numpy as np
import pandas as pd
import torch

from shopfeed.config import ModelConfig
from shopfeed.retrieval.ann import BruteForceIndex
from shopfeed.retrieval.two_tower import in_batch_softmax_loss, positive_pairs, train_two_tower
from tests.conftest import NOW


def test_two_tower_loss_decreases(world, batch, cfg):
    art = train_two_tower(positive_pairs(world["events"]), batch.users, world["products"], cfg.model, steps=80)
    assert np.mean(art.history[-10:]) < np.mean(art.history[:10])


def test_item_embedding_exists_for_zero_interaction_product(world, trained):
    p = world["products"]
    cold = p[(NOW - p.published_at) < 3600]
    assert len(cold) > 0
    touched = set(world["events"].product_id.dropna())
    assert not set(cold.product_id).intersection(touched)
    vecs = trained["retrieval"].embed_items(cold)
    assert np.isfinite(vecs).all()
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_embeddings_unit_norm(world, trained, batch):
    assert np.allclose(np.linalg.norm(trained["retrieval"].item_vecs, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(trained["retrieval"].embed_users(batch.users), axis=1), 1.0, atol=1e-5)


def test_logq_correction_reduces_popular_item_negative_mass():
    torch.manual_seed(0)
    u = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    v = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    log_q = torch.full((8,), np.log(1 / 8))
    log_q[3] = np.log(0.5)  # item 3 is very popular
    base = torch.softmax(u @ v.T / 0.1, dim=1)[:, 3]
    corrected = torch.softmax((u @ v.T / 0.1) - log_q[None, :], dim=1)[:, 3]
    others = torch.arange(8) != 3
    assert (corrected[others] < base[others]).all()
    assert in_batch_softmax_loss(u, v, log_q, 0.1).ndim == 0


def test_ann_matches_brute_force_numpy():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(100, 16)).astype(np.float32)
    ids = np.array([f"p{i}" for i in range(100)], dtype=object)
    idx = BruteForceIndex().build(ids, vecs)
    q = rng.normal(size=16).astype(np.float32)
    got_ids, got_scores = idx.search(q, 5)
    cos = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)) @ (q / np.linalg.norm(q))
    expected = ids[np.argsort(-cos)[:5]]
    assert list(got_ids) == list(expected)
    assert np.all(np.diff(got_scores) <= 0)


def test_ann_k_larger_than_index_and_empty_index():
    idx = BruteForceIndex().build(np.array(["a", "b"], dtype=object), np.eye(2, dtype=np.float32))
    ids, _ = idx.search(np.array([1.0, 0.0]), 10)
    assert len(ids) == 2
    e_ids, e_scores = BruteForceIndex().search(np.array([1.0]), 3)
    assert len(e_ids) == 0 and len(e_scores) == 0


def test_ann_mask_excludes_items():
    idx = BruteForceIndex().build(np.array(["a", "b", "c"], dtype=object), np.eye(3, dtype=np.float32))
    ids, _ = idx.search(np.array([1.0, 0.0, 0.0]), 3, mask=np.array([False, True, True]))
    assert "a" not in ids and len(ids) == 2
