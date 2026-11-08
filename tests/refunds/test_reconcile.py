from refunds.model import Outcome, RefundStatus
from refunds.money import Money
from refunds.provider import FakeProvider
from refunds.reconcile import reconcile_unknown
from tests.refunds.conftest import make_service, order, request


def _timed_out(executed):
    provider = FakeProvider("timeout", executed_before_timeout=executed)
    svc = make_service([order(paid="100.00")], provider=provider)
    res = svc.handle(request(amount="60.00"))
    assert res.decision.outcome is Outcome.APPROVE and res.refund.status is RefundStatus.UNKNOWN
    return svc, provider, res


def test_timeout_holds_money_and_blocks_over_refund():
    svc, provider, _ = _timed_out(executed=True)
    v = svc.ledger.view("o_1")
    assert str(v.reserved) == "60.00 USD" and str(v.remaining) == "40.00 USD"
    other = svc.handle(request("r_2", amount="50.00"))
    assert str(other.refund.amount) == "40.00 USD" and other.refund.status is RefundStatus.UNKNOWN
    assert str(svc.ledger.view("o_1").remaining) == "0.00 USD"


def test_retry_of_timed_out_request_does_not_call_provider_again():
    svc, provider, first = _timed_out(executed=False)
    again = svc.handle(request(amount="60.00"))
    assert again is first and len(provider.calls) == 1


def test_reconcile_money_moved_settles_without_second_payment():
    svc, provider, res = _timed_out(executed=True)
    provider.mode = "ok"
    rep = reconcile_unknown(svc.ledger, provider)
    assert rep.settled == [res.refund.refund_id]
    assert res.refund.status is RefundStatus.SETTLED and res.refund.provider_ref == "prov_000001"
    assert len(provider._settled) == 1


def test_reconcile_money_not_moved_reissues_with_same_key():
    svc, provider, res = _timed_out(executed=False)
    provider.mode = "ok"
    rep = reconcile_unknown(svc.ledger, provider)
    assert rep.settled == [res.refund.refund_id] and res.refund.status is RefundStatus.SETTLED
    assert list(provider._settled) == ["r_1"] and str(svc.ledger.view("o_1").settled) == "60.00 USD"


def test_reconcile_declined_releases_and_timeout_again_keeps_hold():
    svc, provider, res = _timed_out(executed=False)
    rep = reconcile_unknown(svc.ledger, provider)       # still timing out
    assert rep.still_unknown == [res.refund.refund_id] and res.refund.status is RefundStatus.UNKNOWN
    provider.mode = "decline"
    rep = reconcile_unknown(svc.ledger, provider)
    assert rep.failed == [res.refund.refund_id] and res.refund.status is RefundStatus.FAILED
    assert str(svc.ledger.view("o_1").remaining) == "100.00 USD"
