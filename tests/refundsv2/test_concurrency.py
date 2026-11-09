import threading

from refundsv2.db import connect
from refundsv2.ledger import insert_order, ledger_view, list_refunds
from refundsv2.model import RefundStatus
from refundsv2.provider import FakeProvider
from refundsv2.service import RefundService
from tests.refundsv2.conftest import NOW, make_order, req


def test_two_instances_racing_one_order_cap_the_loser(tmp_path):
    """Two service instances (own connections, own provider clients) on one database, both asking 60 of 100."""
    path = str(tmp_path / "shared.db")
    boot = connect(path); insert_order(boot, make_order(paid=10_000)); RefundService(boot, FakeProvider(), now=NOW); boot.close()
    results = {}

    def instance(name, rid):
        svc = RefundService(connect(path), FakeProvider(), now=NOW)
        results[name] = svc.handle(req(rid, 6_000), NOW)

    threads = [threading.Thread(target=instance, args=(f"i{k}", f"r{k}")) for k in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()
    amounts = sorted(r.amount_cents for r in results.values())
    assert amounts == [4_000, 6_000]
    assert all(r.status == RefundStatus.SETTLED for r in results.values())
    c = connect(path)
    assert ledger_view(c, "o1").held_cents == 10_000
    loser = next(r for r in results.values() if r.amount_cents == 4_000)
    assert "capped_to_remaining:40.00" in loser.decision.reasons


def test_many_instances_never_over_refund(tmp_path):
    path = str(tmp_path / "shared.db")
    boot = connect(path); insert_order(boot, make_order(paid=10_000)); RefundService(boot, FakeProvider(), now=NOW); boot.close()

    def instance(k):
        RefundService(connect(path), FakeProvider(), now=NOW).handle(req(f"r{k}", 3_000), NOW)

    threads = [threading.Thread(target=instance, args=(k,)) for k in range(12)]
    for t in threads: t.start()
    for t in threads: t.join()
    c = connect(path)
    rows = list_refunds(c, "o1")
    assert ledger_view(c, "o1").held_cents <= 10_000
    assert sum(r.amount_cents for r in rows if r.status == RefundStatus.SETTLED) == 10_000
    assert sum(1 for r in rows if r.status == RefundStatus.REJECTED) == 12 - sum(1 for r in rows if r.status == RefundStatus.SETTLED)
