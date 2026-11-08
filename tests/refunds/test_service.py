import pytest

from refunds.idempotency import IdempotencyConflict
from refunds.model import Outcome, RefundReason, RefundStatus
from refunds.provider import FakeProvider
from refunds.service import OrderNotFound
from tests.refunds.conftest import make_service, order, request


def test_approved_refund_is_reserved_paid_and_settled(service, provider):
    res = service.handle(request())
    assert res.decision.outcome is Outcome.APPROVE
    assert res.refund.status is RefundStatus.SETTLED and res.refund.provider_ref
    assert provider.calls["r_1"] == request().amount
    assert str(service.ledger.view("o_1").settled) == "40.00 USD"


def test_retry_same_request_returns_same_result_without_second_payment(service, provider):
    first = service.handle(request())
    second = service.handle(request())
    assert second is first
    assert len(provider.calls) == 1 and len(service.ledger.refunds_for("o_1")) == 1


def test_retry_with_different_payload_conflicts(service):
    service.handle(request())
    with pytest.raises(IdempotencyConflict):
        service.handle(request(amount="41.00"))


def test_lost_idempotency_row_still_cannot_double_pay(service, provider):
    service.handle(request())
    service.idempotency._rows.clear()          # simulate crash before the idempotency write
    res = service.handle(request())
    assert res.refund.status is RefundStatus.SETTLED
    assert len(provider.calls) == 1 and str(service.ledger.view("o_1").settled) == "40.00 USD"


def test_distinct_requests_never_exceed_paid(service, provider):
    r1 = service.handle(request("a", amount="70.00"))
    r2 = service.handle(request("b", amount="70.00"))     # capped to 30
    r3 = service.handle(request("c", amount="1.00"))      # nothing left
    assert r1.decision.approved and r2.decision.approved and str(r2.refund.amount) == "30.00 USD"
    assert r3.decision.outcome is Outcome.REJECT and "nothing_left_to_refund" in r3.decision.reasons
    assert str(service.ledger.view("o_1").settled) == "100.00 USD"
    assert sum((m for m in provider.calls.values()), request().amount.zero()) == service.ledger.view("o_1").paid


def test_escalation_moves_no_money(service, provider):
    res = service.handle(request(reason=RefundReason.OTHER))
    assert res.decision.outcome is Outcome.ESCALATE and res.refund is None and not provider.calls


def test_provider_decline_releases_reservation():
    svc = make_service([order()], provider=FakeProvider("decline"))
    res = svc.handle(request())
    assert res.decision.outcome is Outcome.REJECT and res.refund.status is RefundStatus.FAILED
    assert any(r.startswith("provider_declined") for r in res.decision.reasons)
    assert str(svc.ledger.view("o_1").remaining) == "100.00 USD"


def test_history_signal_counts_customer_refunds_across_orders(service):
    service.handle(request("a", "o_1", "10.00"))
    service.handle(request("b", "o_2", "10.00"))
    service.handle(request("c", "o_3", "10.00"))
    res = service.handle(request("d", "o_2", "10.00"))
    assert res.decision.outcome is Outcome.ESCALATE and res.decision.reasons[0].startswith("history_threshold:3>=3")


def test_unknown_order(service):
    with pytest.raises(OrderNotFound):
        service.handle(request(order_id="ghost"))
