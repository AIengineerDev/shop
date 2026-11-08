import pytest

from refunds.ledger import LedgerError, OverRefund, RefundLedger, UnknownOrder
from refunds.model import RefundStatus
from refunds.money import Money
from tests.refunds.conftest import NOW


@pytest.fixture()
def ledger():
    l = RefundLedger()
    l.open_account("o_1", Money.of("100.00"))
    return l


def test_partials_up_to_paid_then_over_refund_raises(ledger):
    for i in range(3):
        r = ledger.reserve("o_1", f"r_{i}", Money.of("33.33"), "v1", NOW)
        ledger.settle(r.refund_id, f"p{i}")
    assert str(ledger.view("o_1").remaining) == "0.01 USD"
    ledger.settle(ledger.reserve("o_1", "r_last", Money.of("0.01"), "v1", NOW).refund_id, "p3")
    with pytest.raises(OverRefund):
        ledger.reserve("o_1", "r_over", Money.of("0.01"), "v1", NOW)


def test_pending_and_unknown_count_against_paid_failed_does_not(ledger):
    a = ledger.reserve("o_1", "a", Money.of("60.00"), "v1", NOW)          # PENDING
    with pytest.raises(OverRefund):
        ledger.reserve("o_1", "b", Money.of("50.00"), "v1", NOW)
    ledger.mark_unknown(a.refund_id)                                       # still held
    with pytest.raises(OverRefund):
        ledger.reserve("o_1", "b", Money.of("50.00"), "v1", NOW)
    c = ledger.reserve("o_1", "c", Money.of("40.00"), "v1", NOW)
    ledger.fail(c.refund_id)                                               # released
    ledger.reserve("o_1", "d", Money.of("40.00"), "v1", NOW)
    v = ledger.view("o_1")
    assert str(v.reserved) == "100.00 USD" and str(v.remaining) == "0.00 USD"


def test_reserve_is_idempotent_on_request_id(ledger):
    a = ledger.reserve("o_1", "r", Money.of("10.00"), "v1", NOW)
    b = ledger.reserve("o_1", "r", Money.of("10.00"), "v1", NOW)
    assert a is b and len(ledger.refunds_for("o_1")) == 1


def test_transitions_are_guarded(ledger):
    r = ledger.reserve("o_1", "r", Money.of("10.00"), "v1", NOW)
    ledger.settle(r.refund_id, "p")
    with pytest.raises(LedgerError):
        ledger.fail(r.refund_id)
    assert r.status is RefundStatus.SETTLED and r.provider_ref == "p"


def test_unknown_order_and_non_positive(ledger):
    with pytest.raises(UnknownOrder):
        ledger.view("nope")
    with pytest.raises(LedgerError):
        ledger.reserve("o_1", "z", Money.zero(), "v1", NOW)
