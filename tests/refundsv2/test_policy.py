import pytest

from refundsv2.model import LedgerView, Order, OrderStatus, Outcome, Reason, RefundRequest, Signals
from refundsv2.policy import DAY, DEFAULT_POLICY, RULES, PolicyConfig, decide

NOW = 1_000_000.0


def order(**over):
    base = dict(order_id="o1", customer_id="c1", paid_cents=10_000, currency="USD", status=OrderStatus.DELIVERED,
                paid_at=NOW - 10 * DAY, shipped_at=NOW - 8 * DAY, delivered_at=NOW - 5 * DAY)
    base.update(over)
    return Order(**base)


def request(amount=4_000, reason=Reason.DAMAGED, currency="USD"):
    return RefundRequest("r1", "o1", amount, currency, reason, NOW)


def run(o=None, r=None, held=0, signals=Signals(), policy=DEFAULT_POLICY, now=NOW):
    o = o or order()
    return decide(o, r or request(), LedgerView(o.paid_cents, held), signals, policy, now)


def test_happy_path_approves_full_amount_with_complete_trace():
    d = run()
    assert d.outcome == Outcome.APPROVE and d.amount_cents == 4_000 and d.policy_version == "v1"
    assert [t.name for t in d.trace] == [n for n, _ in RULES]
    assert all(t.passed for t in d.trace) and d.reasons == ("all_rules_passed",)


@pytest.mark.parametrize("status", [OrderStatus.CREATED, OrderStatus.CANCELLED])
def test_non_refundable_status_rejects_and_stops(status):
    d = run(order(status=status))
    assert d.outcome == Outcome.REJECT and d.reasons == (f"order_not_refundable:status={status.value}",)
    assert len(d.trace) == 1  # reject short-circuits


def test_amount_and_currency_validation():
    assert run(r=request(amount=0)).reasons == ("amount_invalid:must_be_positive",)
    assert run(r=request(currency="EUR")).reasons == ("currency_mismatch:EUR!=USD",)


def test_cap_to_remaining_and_nothing_left():
    d = run(r=request(amount=8_000), held=7_000)
    assert d.outcome == Outcome.APPROVE and d.amount_cents == 3_000 and "capped_to_remaining:30.00" in d.reasons
    d = run(held=10_000)
    assert d.outcome == Outcome.REJECT and d.reasons == ("nothing_left_to_refund",)


def test_window_is_delivery_anchored_and_inclusive():
    exactly = order(delivered_at=NOW - 30 * DAY)
    assert run(exactly).outcome == Outcome.APPROVE
    late = order(delivered_at=NOW - 30 * DAY - 1)
    d = run(late)
    assert d.outcome == Outcome.ESCALATE and "outside_window:30d" in d.reasons
    # no delivery -> anchored on payment
    unshipped = order(status=OrderStatus.PAID, shipped_at=None, delivered_at=None, paid_at=NOW - 31 * DAY)
    assert "outside_window:30d" in run(unshipped).reasons


def test_changed_mind_only_before_shipment_and_other_always_reviewed():
    paid = order(status=OrderStatus.PAID, shipped_at=None, delivered_at=None)
    assert run(paid, request(reason=Reason.CHANGED_MIND)).outcome == Outcome.APPROVE
    d = run(request_shipped := order(), request(reason=Reason.CHANGED_MIND))
    assert d.outcome == Outcome.ESCALATE and "reason_requires_review:changed_mind" in d.reasons
    assert "reason_requires_review:other" in run(r=request(reason=Reason.OTHER)).reasons


def test_auto_limit_applies_to_capped_amount():
    big = order(paid_cents=50_000)
    d = run(big, request(amount=30_000))
    assert d.outcome == Outcome.ESCALATE and "above_auto_limit:200.00" in d.reasons
    # requested 300 but only 150 left -> capped under the limit -> approve
    d = run(big, request(amount=30_000), held=35_000)
    assert d.outcome == Outcome.APPROVE and d.amount_cents == 15_000


def test_history_threshold():
    assert run(signals=Signals(refunds_last_90d=2)).outcome == Outcome.APPROVE
    d = run(signals=Signals(refunds_last_90d=3))
    assert d.outcome == Outcome.ESCALATE and "history_threshold:3>=3" in d.reasons


def test_abuse_score_is_bounded():
    assert run(signals=Signals(abuse_score=None)).outcome == Outcome.APPROVE
    assert run(signals=Signals(abuse_score=0.3)).outcome == Outcome.APPROVE
    d = run(signals=Signals(abuse_score=0.9))
    assert d.outcome == Outcome.ESCALATE and d.reasons == ("abuse_score:0.90>=0.80",)
    # 0.6 halves the limit to 100: a 40 request still passes, a 150 request now escalates
    assert run(signals=Signals(abuse_score=0.6)).outcome == Outcome.APPROVE
    d = run(order(paid_cents=50_000), request(amount=15_000), signals=Signals(abuse_score=0.6))
    assert d.outcome == Outcome.ESCALATE and "above_auto_limit:100.00" in d.reasons
    assert any(r.startswith("abuse_score:0.60>=0.50:limit=100.00") for r in d.reasons)


def test_escalate_collects_every_failing_reason():
    d = run(order(delivered_at=NOW - 40 * DAY), request(amount=9_000, reason=Reason.OTHER), signals=Signals(refunds_last_90d=5))
    assert d.outcome == Outcome.ESCALATE
    assert set(d.reasons) >= {"outside_window:30d", "reason_requires_review:other", "history_threshold:5>=3"}
    assert d.amount_cents == 9_000  # what a reviewer would be approving


def test_policy_thresholds_are_data():
    strict = PolicyConfig(version="v2", window_days=3, auto_limit_cents=1_000)
    d = run(policy=strict)
    assert d.policy_version == "v2" and d.outcome == Outcome.ESCALATE
    assert {"outside_window:3d", "above_auto_limit:10.00"} <= set(d.reasons)


def test_decision_is_pure():
    a, b = run(), run()
    assert a == b
