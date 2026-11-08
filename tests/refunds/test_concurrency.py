"""Two requests for the same order land on two instances at once: exactly the paid amount leaves."""
import threading

import pytest

from refunds.ledger import ConcurrencyConflict, OverRefund, RefundLedger
from refunds.model import Outcome
from refunds.money import Money
from refunds.provider import FakeProvider
from refunds.service import OrderRepo, RefundService
from refunds.config import PolicyConfig
from tests.refunds.conftest import NOW, order, request


def test_stale_version_is_refused():
    l = RefundLedger()
    l.open_account("o", Money.of("100.00"))
    v0 = l.view("o").version
    l.reserve("o", "a", Money.of("10.00"), "v1", NOW, expected_version=v0)
    with pytest.raises(ConcurrencyConflict):
        l.reserve("o", "b", Money.of("10.00"), "v1", NOW, expected_version=v0)
    l.reserve("o", "b", Money.of("10.00"), "v1", NOW, expected_version=l.view("o").version)


def test_ledger_race_many_threads_never_exceeds_paid():
    l = RefundLedger()
    l.open_account("o", Money.of("100.00"))
    wins, start = [], threading.Barrier(16)

    def worker(i):
        start.wait()
        try:
            l.reserve("o", f"r{i}", Money.of("30.00"), "v1", NOW)
            wins.append(i)
        except OverRefund:
            pass

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(wins) == 3 and str(l.view("o").reserved) == "90.00 USD"


def test_two_instances_shared_ledger_one_order():
    """Two RefundService instances (two app servers) sharing one ledger and provider."""
    ledger, provider = RefundLedger(), FakeProvider("ok")
    orders = [order(paid="100.00")]
    a = RefundService(OrderRepo(orders), ledger, provider, PolicyConfig(), clock=lambda: NOW)
    b = RefundService(OrderRepo(orders), ledger, provider, PolicyConfig(), clock=lambda: NOW)
    results, start = {}, threading.Barrier(2)

    def run(svc, rid):
        start.wait()
        results[rid] = svc.handle(request(rid, amount="70.00"))

    ta, tb = threading.Thread(target=run, args=(a, "A")), threading.Thread(target=run, args=(b, "B"))
    ta.start(); tb.start(); ta.join(); tb.join()
    amounts = sorted(str(r.refund.amount) for r in results.values() if r.refund)
    assert amounts == ["30.00 USD", "70.00 USD"]   # second was re-decided and capped, not rejected
    assert all(r.decision.outcome is Outcome.APPROVE for r in results.values())
    assert str(ledger.view("o_1").settled) == "100.00 USD"
    assert sum((m for m in provider.calls.values()), Money.zero()) == Money.of("100.00")
