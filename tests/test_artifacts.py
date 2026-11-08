"""Save/load round trip: the serving path must reproduce training-time predictions exactly."""
import numpy as np
import pytest

from shopfeed.ranking.light import LightRanker
from shopfeed.training.train_ranker import load_artifacts, save_artifacts


def test_round_trip_reproduces_predictions(tmp_path, trained, batch, world):
    save_artifacts(tmp_path, trained["retrieval"], trained["light"], trained["heavy"])
    assert not (tmp_path / "light.pkl").exists() and (tmp_path / "light.npz").exists()
    r2, l2, h2 = load_artifacts(tmp_path)
    X = trained["ts"].X[:500]
    np.testing.assert_allclose(l2.score(X), trained["light"].score(X), atol=1e-6)
    np.testing.assert_allclose(h2.predict_proba(X), trained["heavy"].predict_proba(X), atol=1e-6)
    np.testing.assert_allclose(r2.embed_users(batch.users.head(20)), trained["retrieval"].embed_users(batch.users.head(20)), atol=1e-6)
    np.testing.assert_allclose(r2.item_vecs, trained["retrieval"].item_vecs, atol=1e-6)
    assert h2.version == trained["heavy"].version


def test_light_numpy_inference_matches_sklearn(trained):
    X = trained["ts"].X[:300]
    light = trained["light"]
    from shopfeed.ranking.light import LIGHT_IDX
    from shopfeed.ranking.position import neutralize_position
    expected = light.clf.predict_proba(neutralize_position(X)[:, LIGHT_IDX])[:, 1]
    np.testing.assert_allclose(light.score(X), expected, atol=1e-6)


def test_unfitted_light_cannot_be_saved_and_feature_drift_is_rejected(trained):
    with pytest.raises(ValueError):
        LightRanker().to_state()
    state = trained["light"].to_state()
    state["features"] = np.array(["something_else"])
    with pytest.raises(ValueError, match="retrain"):
        LightRanker.from_state(state)
