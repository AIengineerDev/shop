import numpy as np
import pandas as pd
import pytest
import torch

from shopfeed.config import HEADS, ModelConfig, ValueWeights
from shopfeed.features.snapshot import N_FEATURES, POSITION_IDX
from shopfeed.ranking.heavy import MultiTaskRanker, train_ranker
from shopfeed.ranking.light import LightRanker
from shopfeed.ranking.position import neutralize_position
from shopfeed.ranking.value import value_score


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
    Y = (X[:, :len(HEADS)] > 0.3).astype(np.float32)
    return X, Y


def test_heavy_forward_shape_and_range():
    X, _ = _toy()
    m = MultiTaskRanker(N_FEATURES, 16, 2)
    m.eval()
    with torch.no_grad():
        out = torch.sigmoid(m(torch.from_numpy(X)))
    assert out.shape == (400, len(HEADS))
    assert (out > 0).all() and (out < 1).all()


def test_heavy_loss_decreases_on_toy():
    X, Y = _toy()
    art = train_ranker(X, Y, ModelConfig(hidden=32, lr=1e-2, batch_size=128), steps=150)
    assert np.mean(art.history[-10:]) < np.mean(art.history[:10])


def test_gradient_flows_to_every_head():
    X, Y = _toy(64)
    m = MultiTaskRanker(N_FEATURES, 16, 2)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(m(torch.from_numpy(X)), torch.from_numpy(Y))
    loss.backward()
    for tower in m.towers:
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in tower.parameters())


def test_empty_training_set_raises():
    with pytest.raises(ValueError, match="empty"):
        train_ranker(np.zeros((0, N_FEATURES)), np.zeros((0, len(HEADS))), ModelConfig())


def test_predict_ignores_position_but_training_sees_it():
    X, Y = _toy()
    art = train_ranker(X, Y, ModelConfig(hidden=32), steps=30)
    X0, X19 = X.copy(), X.copy()
    X0[:, POSITION_IDX] = 0
    X19[:, POSITION_IDX] = 19
    np.testing.assert_array_equal(art.predict_proba(X0), art.predict_proba(X19))
    art.model.eval()
    with torch.no_grad():
        raw0 = art.model(torch.from_numpy(X0))
        raw19 = art.model(torch.from_numpy(X19))
    assert not torch.allclose(raw0, raw19)
    assert neutralize_position(X19)[:, POSITION_IDX].sum() == 0 and X19[:, POSITION_IDX].sum() > 0  # copy, not in place


def test_light_cut_preserves_ids_and_handles_small_pool():
    X, Y = _toy()
    lr = LightRanker().fit(X, Y[:, 0].astype(int))
    cands = pd.DataFrame({"product_id": [f"p{i}" for i in range(400)]})
    cut = lr.cut(cands, X, 50)
    assert len(cut) == 50 and set(cut.product_id) <= set(cands.product_id)
    assert len(lr.cut(cands.head(10), X[:10], 50)) == 10
    assert cut.light_score.is_monotonic_decreasing


def test_light_untrained_fallback_scores():
    X, _ = _toy()
    assert LightRanker().score(X).shape == (400,)


def test_light_requires_two_classes():
    X, _ = _toy()
    with pytest.raises(ValueError):
        LightRanker().fit(X, np.zeros(400, dtype=int))


# ---------------------------------------------------------------- value
def _probs(**over):
    p = {h: 0.05 for h in HEADS}
    p.update(over)
    return np.array([[p[h] for h in HEADS]])


def test_hide_heavy_item_scores_below_neutral_with_equal_purchase_prob():
    w = ValueWeights()
    neutral = value_score(_probs(purchase=0.1, hide=0.01), np.array([50.0]), w)[0]
    hidey = value_score(_probs(purchase=0.1, hide=0.5), np.array([50.0]), w)[0]
    assert hidey < neutral


def test_zero_weights_give_zero_value():
    w = ValueWeights(click=0, dwell=0, save=0, atc=0, purchase=0, hide=0, return_=0)
    assert value_score(_probs(purchase=0.9), np.array([50.0]), w)[0] == 0.0


def test_changing_weights_reorders_without_retraining():
    P = np.vstack([_probs(purchase=0.3, hide=0.4), _probs(purchase=0.2, hide=0.01)])
    prices = np.array([50.0, 50.0])
    punish = value_score(P, prices, ValueWeights(hide=-2.0))
    ignore = value_score(P, prices, ValueWeights(hide=0.0))
    assert np.argmax(punish) == 1 and np.argmax(ignore) == 0


def test_order_value_scales_purchase_but_is_clipped():
    w = ValueWeights(click=0, dwell=0, save=0, atc=0, purchase=1, hide=0, return_=0)
    cheap = value_score(_probs(purchase=0.1), np.array([50.0]), w)[0]
    pricey = value_score(_probs(purchase=0.1), np.array([500.0]), w)[0]
    absurd = value_score(_probs(purchase=0.1), np.array([50_000.0]), w)[0]
    assert cheap < pricey == absurd


def test_value_rejects_wrong_shape():
    with pytest.raises(ValueError):
        value_score(np.zeros((3, 2)), np.ones(3), ValueWeights())
