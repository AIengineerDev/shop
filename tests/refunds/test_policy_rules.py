"""One pass + one fail test per rule, with literals only: decide() is pure."""
from datetime import timedelta

from refunds.config import PolicyConfig
from refunds.model import OrderStatus, Outcome, RefundReason, Signals
from refunds.policy import decide
from tests.refunds.conftest import DAY, NOW, order, request, view

P = PolicyConfig()


def run(o=None, r=None, v=None, now=NOW, signals=Signals(), policy=P):
    return decide(o or order(), r or request(), v or view(), now, policy, signals)


def test_happy_path_approves_requested_amount():
    d = run()
    assert d.outcome is Outcome.APPROVE and d.amount == request().amount
    assert d.reasons == ["eligible"] and d.policy_version == "v1"


def test_unpaid_or_cancelled_order_rejected():
    for st in (OrderStatus.PENDING_PAYMENT, OrderStatus.CANCELLED):
        d = run(order(status=st, delivered_at=None))
        assert d.outcome is Outcome.REJECT
        assert any(r.startswith("order_not_refundable") for r in d.reasons)


def test_zero_or_negative_amount_rejected():
    for amt in ("0.00", "-5.00"):
        d = run(r=request(amount=amt))
        assert d.outcome is Outcome.REJECT and any(r.startswith("amount_invalid") for r in d.reasons)


def test_currency_mismatch_rejected():
    r = request()
    r = type(r)(r.request_id, r.order_id, r.amount.__class__.of("40", "EUR"), r.reason, r.submitted_at)
    d = run(r=r)
    assert d.outcome is Outcome.REJECT and any(r.startswith("currency_mismatch") for r in d.reasons)


def test_amount_above_remaining_is_capped_not_rejected():
    d = run(r=request(amount="80.00"), v=view(paid="100.00", settled="30.00", reserved="10.00"))
    assert d.outcome is Outcome.APPROVE
    assert str(d.amount) == "60.00 USD"
    assert d.reasons == ["capped_to_remaining:60.00 USD"]


def test_nothing_left_rejects():
    d = run(v=view(paid="100.00", settled="100.00"))
    assert d.outcome is Outcome.REJECT and "nothing_left_to_refund" in d.reasons


def test_window_anchored_on_delivery_then_payment():
    late = NOW + 31 * DAY
    d = run(now=late)
    assert d.outcome is Outcome.ESCALATE and "outside_window:30d" in d.reasons
    # not delivered: anchor is paid_at (5 days ago) -> 30 days from then is still inside at NOW+20d
    d = run(order(status=OrderStatus.SHIPPED), now=NOW + 20 * DAY)
    assert d.outcome is Outcome.APPROVE
    d = run(order(status=OrderStatus.SHIPPED), now=NOW + 26 * DAY)
    assert d.outcome is Outcome.ESCALATE


def test_window_boundary_inclusive():
    d = run(now=order().delivered_at + timedelta(days=30))
    assert d.outcome is Outcome.APPROVE


def test_reason_other_escalates():
    d = run(r=request(reason=RefundReason.OTHER))
    assert d.outcome is Outcome.ESCALATE and "reason_requires_review:other" in d.reasons


def test_changed_mind_only_before_shipment():
    assert run(order(status=OrderStatus.PAID), request(reason=RefundReason.CHANGED_MIND)).outcome is Outcome.APPROVE
    d = run(order(status=OrderStatus.SHIPPED), request(reason=RefundReason.CHANGED_MIND))
    assert d.outcome is Outcome.ESCALATE and "reason_requires_review:changed_mind_after_shipment" in d.reasons


def test_above_auto_limit_escalates_and_uses_capped_amount():
    d = run(order(paid="500.00"), request(amount="250.00"), view(paid="500.00"))
    assert d.outcome is Outcome.ESCALATE and "above_auto_limit:200.00 USD" in d.reasons
    # requested 250 but only 150 remains -> capped to 150 -> under the limit -> approved
    d = run(order(paid="500.00"), request(amount="250.00"), view(paid="500.00", settled="350.00"))
    assert d.outcome is Outcome.APPROVE and str(d.amount) == "150.00 USD"


def test_history_threshold_escalates():
    d = run(signals=Signals(customer_refunds_in_window=3))
    assert d.outcome is Outcome.ESCALATE and d.reasons == ["history_threshold:3>=3_in_90d"]
    assert run(signals=Signals(customer_refunds_in_window=2)).outcome is Outcome.APPROVE


def test_reject_short_circuits():
    d = run(order(status=OrderStatus.CANCELLED, delivered_at=None), request(amount="-1", reason=RefundReason.OTHER))
    assert d.outcome is Outcome.REJECT and len(d.reasons) == 1


def test_escalation_lists_every_failing_reason():
    d = run(order(paid="500.00"), request(amount="300.00", reason=RefundReason.OTHER), view(paid="500.00"),
            now=NOW + 40 * DAY, signals=Signals(customer_refunds_in_window=5))
    assert d.outcome is Outcome.ESCALATE
    assert [r.split(":")[0] for r in d.reasons] == ["outside_window", "reason_requires_review", "above_auto_limit", "history_threshold"]


def test_policy_thresholds_are_data():
    strict = PolicyConfig(version="v2", auto_limit="20.00")
    d = run(policy=strict)
    assert d.outcome is Outcome.ESCALATE and d.policy_version == "v2"
