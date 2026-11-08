import numpy as np
import pandas as pd
import pytest

from shopfeed.config import CATEGORIES, HOUR
from shopfeed.features.batch import AFF_COLS, compute_batch_state
from shopfeed.features.snapshot import FEATURE_NAMES, N_FEATURES, POSITION_IDX, build_features
from shopfeed.features.streaming import StreamingStore
from tests.conftest import NOW


# ---------------------------------------------------------------- batch
def test_affinity_sums_to_one(batch):
    s = batch.users[AFF_COLS].sum(axis=1).values
    assert np.allclose(s, 1.0, atol=1e-6)


def test_user_without_engagement_gets_uniform_prior(world, batch):
    cold = world["users"][world["users"].cohort == "cold"].user_id.iloc[0]
    row = batch.users.set_index("user_id").loc[cold]
    assert np.allclose(row[AFF_COLS].values.astype(float), 1.0 / len(CATEGORIES))
    assert row.n_purchases == 0


def test_merchant_quality_bounded(batch):
    assert batch.merchants.return_rate.between(0, 1).all()
    assert batch.merchants.review_score.between(0, 5).all()
    assert batch.merchants.is_suspended.isin([0.0, 1.0]).all()


def test_batch_state_ignores_events_at_or_after_as_of(world):
    ev = world["events"]
    first_purchase_ts = ev[ev.type == "purchase"].ts.min()
    state = compute_batch_state(ev, world["impressions"], world["users"], world["products"], world["merchants"], first_purchase_ts)
    assert state.users.n_purchases.sum() == 0
    assert state.products.n_purchases.sum() == 0


# ---------------------------------------------------------------- streaming
def _store(rows):
    s = StreamingStore()
    s.add_events(pd.DataFrame(rows, columns=["event_id", "user_id", "product_id", "type", "ts"]))
    return s


def test_window_boundary_is_half_open():
    s = _store([("e1", "u", "p", "view", 100.0), ("e2", "u", "p", "view", 200.0)])
    assert s.product_count("view", ["p"], np.array([200.0]), window_s=100)[0] == 1  # (100, 200]
    assert s.product_count("view", ["p"], np.array([200.0]), window_s=101)[0] == 2
    assert s.product_count("view", ["p"], np.array([99.0]), window_s=100)[0] == 0


def test_future_events_ignored_and_duplicates_dropped():
    s = StreamingStore()
    df = pd.DataFrame([("e1", "u", "p", "view", 100.0), ("e1", "u", "p", "view", 100.0), ("e2", "u", "p", "view", 500.0)],
                      columns=["event_id", "user_id", "product_id", "type", "ts"])
    res = s.add_events(df, now=150.0)
    assert res == {"accepted": 1, "duplicates": 1, "ignored_future": 1}
    assert s.product_count_now("view", ["p"], 150.0, HOUR)[0] == 1


def test_empty_store_returns_zeros():
    s = StreamingStore()
    out = s.product_count("purchase", ["a", "b"], np.array([1.0, 2.0]))
    assert out.tolist() == [0.0, 0.0]
    assert s.user_last_ts("nobody", "purchase") == {}


# ---------------------------------------------------------------- snapshot
def _pairs(world, n=50, ts=NOW, position=0):
    imp = world["impressions"].head(n)
    return pd.DataFrame({"user_id": imp.user_id.values, "product_id": imp.product_id.values, "ts": ts, "position": position})


def test_snapshot_is_deterministic_and_fixed_width(world, batch, streaming):
    X1 = build_features(_pairs(world), batch, streaming, world["products"], world["follows"])
    X2 = build_features(_pairs(world), batch, streaming, world["products"], world["follows"])
    assert X1.shape == (50, N_FEATURES)
    assert len(FEATURE_NAMES) == N_FEATURES and FEATURE_NAMES[POSITION_IDX] == "position"
    np.testing.assert_array_equal(X1, X2)


def test_snapshot_at_time_t_excludes_events_after_t(world, batch):
    pid = world["products"].product_id.iloc[0]
    s = StreamingStore()
    s.add_events(pd.DataFrame([("e1", "u_0", pid, "view", NOW + 1.0)], columns=["event_id", "user_id", "product_id", "type", "ts"]))
    pairs = pd.DataFrame({"user_id": ["u_0"], "product_id": [pid], "ts": [NOW], "position": [0]})
    X_before = build_features(pairs, batch, s, world["products"], world["follows"])
    X_after = build_features(pairs.assign(ts=NOW + 2.0), batch, s, world["products"], world["follows"])
    i = FEATURE_NAMES.index("p_log_views_1h")
    assert X_before[0, i] == 0.0
    assert X_after[0, i] == pytest.approx(np.log1p(1))


def test_snapshot_fills_missing_user_with_defaults_never_nan(world, batch, streaming):
    pairs = pd.DataFrame({"user_id": ["ghost"], "product_id": [world["products"].product_id.iloc[0]], "ts": [NOW], "position": [0]})
    X = build_features(pairs, batch, streaming, world["products"], world["follows"])
    assert not np.isnan(X).any()
    assert X[0, :len(CATEGORIES)].sum() == pytest.approx(1.0)  # uniform prior affinity


def test_snapshot_empty_input(world, batch, streaming):
    X = build_features(_pairs(world, n=0), batch, streaming, world["products"], world["follows"])
    assert X.shape == (0, N_FEATURES)
