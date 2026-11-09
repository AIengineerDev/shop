import threading

import pytest

from refundsv2.db import connect
from refundsv2.ledger import (DuplicateRequest, InsufficientFunds, UnknownOrder, count_customer_refunds_since, get_refund,
                              insert_order, ledger_view, list_refunds, record_decision, reserve, transition)
from refundsv2.model import Decision, Order, OrderStatus, Outcome, Reason, RefundRequest, RefundStatus

NOW = 1_000.0


def _order(oid="o1", paid=9_999, customer="c1"):
    return Order(oid, customer, paid, "USD", OrderStatus.DELIVERED, NOW - 10, NOW - 8, NOW - 5)


def _req(rid, amount, oid="o1"):
    return RefundRequest(rid, oid, amount, "USD", Reason.DAMAGED, NOW)


def _dec(amount, outcome=Outcome.APPROVE):
    return Decision(outcome, amount, ("all_rules_passed",), (), "v1")


@pytest.fixture
def conn():
    c = connect()
    insert_order(c, _order())
    return c


def test_three_partials_to_the_cent_then_nothing_left(conn):
    for i in range(3):
        reserve(conn, _req(f"r{i}", 3_333), _dec(3_333), NOW)
    assert ledger_view(conn, "o1").remaining_cents == 0
    with pytest.raises(InsufficientFunds):
        reserve(conn, _req("r3", 1), _dec(1), NOW)
    assert len(list_refunds(conn, "o1")) == 3


def test_pending_and_unknown_hold_money_failed_releases_it(conn):
    reserve(conn, _req("a", 6_000), _dec(6_000), NOW)
    with pytest.raises(InsufficientFunds):
        reserve(conn, _req("b", 5_000), _dec(5_000), NOW)
    assert transition(conn, "a", [RefundStatus.PENDING], RefundStatus.UNKNOWN, NOW)
    with pytest.raises(InsufficientFunds):
        reserve(conn, _req("b", 5_000), _dec(5_000), NOW)
    assert transition(conn, "a", [RefundStatus.UNKNOWN], RefundStatus.FAILED, NOW)
    assert ledger_view(conn, "o1").remaining_cents == 9_999
    reserve(conn, _req("b", 5_000), _dec(5_000), NOW)


def test_request_id_is_unique_even_when_money_is_available(conn):
    reserve(conn, _req("same", 100), _dec(100), NOW)
    with pytest.raises(DuplicateRequest):
        reserve(conn, _req("same", 100), _dec(100), NOW)
    with pytest.raises(DuplicateRequest):
        record_decision(conn, _req("same", 100), _dec(0, Outcome.REJECT), RefundStatus.REJECTED, NOW)
    assert ledger_view(conn, "o1").held_cents == 100


def test_recorded_decisions_hold_nothing_but_are_replayable(conn):
    r = record_decision(conn, _req("e", 500), _dec(500, Outcome.ESCALATE), RefundStatus.ESCALATED, NOW)
    assert r.status == RefundStatus.ESCALATED and r.amount_cents == 0 and r.decision.outcome == Outcome.ESCALATE
    assert ledger_view(conn, "o1").held_cents == 0
    assert get_refund(conn, "e").decision.amount_cents == 500


def test_transition_is_guarded(conn):
    reserve(conn, _req("a", 100), _dec(100), NOW)
    assert transition(conn, "a", [RefundStatus.PENDING], RefundStatus.SETTLED, NOW, provider_ref="pr_1")
    assert not transition(conn, "a", [RefundStatus.PENDING], RefundStatus.FAILED, NOW)  # already settled: no-op
    r = get_refund(conn, "a")
    assert r.status == RefundStatus.SETTLED and r.provider_ref == "pr_1"


def test_unknown_order_and_bad_amount(conn):
    with pytest.raises(UnknownOrder):
        reserve(conn, _req("x", 100, oid="nope"), _dec(100), NOW)
    with pytest.raises(ValueError):
        reserve(conn, _req("x", 0), _dec(0), NOW)
    with pytest.raises(UnknownOrder):
        ledger_view(conn, "nope")


def test_customer_history_counts_only_holds(conn):
    insert_order(conn, _order("o2", customer="c1"))
    reserve(conn, _req("a", 100), _dec(100), NOW)
    reserve(conn, _req("b", 100, oid="o2"), _dec(100), NOW)
    record_decision(conn, _req("c", 100), _dec(100, Outcome.ESCALATE), RefundStatus.ESCALATED, NOW)
    transition(conn, "b", [RefundStatus.PENDING], RefundStatus.FAILED, NOW)
    assert count_customer_refunds_since(conn, "c1", NOW - 1) == 1
    assert count_customer_refunds_since(conn, "c1", NOW + 1) == 0


def test_concurrent_reservations_never_exceed_paid(tmp_path):
    """16 threads, each its own connection (an 'instance'), all hitting one order worth 100.00 with 30.00 asks."""
    path = str(tmp_path / "ledger.db")
    boot = connect(path)
    insert_order(boot, _order(paid=10_000))
    boot.close()
    results = []

    def worker(i):
        c = connect(path)
        try:
            reserve(c, _req(f"t{i}", 3_000), _dec(3_000), NOW)
            results.append("ok")
        except InsufficientFunds:
            results.append("insufficient")
        finally:
            c.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads: t.start()
    for t in threads: t.join()
    c = connect(path)
    assert results.count("ok") == 3 and results.count("insufficient") == 13
    assert ledger_view(c, "o1").held_cents == 9_000 <= 10_000
