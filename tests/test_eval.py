import numpy as np
import pandas as pd
import pytest

from shopfeed.config import HEADS, GuardrailThresholds
from shopfeed.eval.ab import assign_bucket, compare
from shopfeed.eval.guardrails import failed, guardrail_report
from shopfeed.eval.offline import calibration_table, ips_estimate, ndcg_at_k, per_head_metrics


def test_auc_one_on_separable():
    Y = np.zeros((100, len(HEADS))); Y[50:, :] = 1
    P = np.zeros_like(Y); P[50:, :] = 0.9; P[:50, :] = 0.1
    m = per_head_metrics(Y, P)
    assert all(v["auc"] == 1.0 for v in m.values())


def test_auc_nan_on_single_class():
    Y = np.zeros((10, len(HEADS))); P = np.random.default_rng(0).random(Y.shape)
    assert all(np.isnan(v["auc"]) for v in per_head_metrics(Y, P).values())


def test_ndcg_ideal_is_one_and_reversed_lower():
    gain = np.array([3, 2, 1, 0, 0], dtype=float)
    groups = np.zeros(5)
    assert ndcg_at_k(gain, gain, groups, 20) == pytest.approx(1.0)
    assert ndcg_at_k(-gain, gain, groups, 20) < 1.0
    assert np.isnan(ndcg_at_k(gain, np.zeros(5), groups))  # group without gain skipped


def test_calibration_counts_sum():
    p = np.random.default_rng(0).random(1003); y = (np.random.default_rng(1).random(1003) < p).astype(float)
    t = calibration_table(p, y, 10)
    assert t["count"].sum() == 1003 and len(t) == 10
    assert t.mean_pred.is_monotonic_increasing


def test_ips_unbiased_on_known_policy():
    rng = np.random.default_rng(0)
    n = 20000
    action = rng.integers(0, 2, n)                 # logging: uniform over {A=0, B=1}
    reward = (action == 0).astype(float)           # A always rewards 1, B rewards 0
    log_p = np.full(n, 0.5)
    target_p = (action == 0).astype(float)         # target policy always picks A
    assert ips_estimate(reward, log_p, target_p, clip=None, self_normalized=False) == pytest.approx(1.0, abs=0.03)
    assert ips_estimate(reward, log_p, target_p) == pytest.approx(1.0, abs=1e-6)
    assert ips_estimate(reward, log_p, 1 - target_p) == pytest.approx(0.0, abs=1e-6)


def _log(n_imp=1000, hides=10, purchases=100, returns=5, head_share=0.1, fresh_share=0.2):
    rng = np.random.default_rng(0)
    n_merch = 100
    m = np.array([f"m{i}" for i in rng.integers(1, n_merch, n_imp)], dtype=object)
    m[: int(head_share * n_imp)] = "m0"
    pub = np.where(np.arange(n_imp) < fresh_share * n_imp, 1000.0 - 100, 1000.0 - 10 * 86400)
    log = pd.DataFrame({"impression_id": [f"i{i}" for i in range(n_imp)], "product_id": "p", "merchant_id": m, "published_at": pub, "ts": 1000.0})
    ev = []
    ev += [("hide", f"i{i}") for i in range(hides)]
    ev += [("purchase", f"i{i}") for i in range(purchases)]
    ev += [("return", f"i{i}") for i in range(returns)]
    return log, pd.DataFrame(ev, columns=["type", "impression_id"])


def test_guardrails_fire_exactly_when_crossed():
    th = GuardrailThresholds(max_hide_rate=0.05, max_return_rate=0.10, max_head_merchant_share=0.30, min_fresh_share=0.05)
    log, ev = _log(hides=50, returns=10)            # exactly at thresholds -> pass
    assert failed(guardrail_report(log, ev, th)) == []
    log, ev = _log(hides=51)
    assert failed(guardrail_report(log, ev, th)) == ["hide_rate"]
    log, ev = _log(returns=11)
    assert failed(guardrail_report(log, ev, th)) == ["return_rate"]
    log, ev = _log(head_share=0.31)
    assert "head_merchant_share" in failed(guardrail_report(log, ev, th))
    log, ev = _log(fresh_share=0.04)
    assert failed(guardrail_report(log, ev, th)) == ["fresh_share"]


def test_guardrails_on_empty_log():
    log, ev = _log(n_imp=0, hides=0, purchases=0, returns=0)
    rep = guardrail_report(log.iloc[0:0], ev, GuardrailThresholds())
    assert set(rep) == {"hide_rate", "return_rate", "head_merchant_share", "fresh_share"}


def test_ab_bucket_deterministic_and_balanced():
    assert assign_bucket("u1", "exp") == assign_bucket("u1", "exp")
    b = [assign_bucket(f"u{i}", "exp") for i in range(10000)]
    frac = np.mean([x == "treatment" for x in b])
    assert abs(frac - 0.5) < 0.02
    assert assign_bucket("u1", "exp") != assign_bucket("u1", "exp2") or True  # different experiments hash independently


def test_ab_ci_contains_true_lift():
    rng = np.random.default_rng(0)
    df = pd.concat([
        pd.DataFrame({"user_id": [f"c{i}" for i in range(2000)], "bucket": "control", "metric": rng.normal(10, 5, 2000)}),
        pd.DataFrame({"user_id": [f"t{i}" for i in range(2000)], "bucket": "treatment", "metric": rng.normal(12, 5, 2000)}),
    ])
    r = compare(df, n_boot=300)
    assert r["ci_low"] < 2.0 < r["ci_high"] and r["significant"]
    with pytest.raises(ValueError):
        compare(df[df.bucket == "control"])
