import pytest

from refundsv2.db import connect
from refundsv2.ledger import insert_order
from refundsv2.model import Order, OrderStatus, Reason, RefundRequest
from refundsv2.policy import DAY
from refundsv2.provider import FakeProvider
from refundsv2.service import RefundService

NOW = 1_700_000_000.0


def make_order(oid="o1", paid=10_000, status=OrderStatus.DELIVERED, customer="c1", **over):
    base = dict(paid_at=NOW - 10 * DAY, shipped_at=NOW - 8 * DAY, delivered_at=NOW - 5 * DAY)
    if status == OrderStatus.PAID:
        base.update(shipped_at=None, delivered_at=None)
    base.update(over)
    return Order(oid, customer, paid, "USD", status, **base)


def req(rid="r1", amount=4_000, oid="o1", reason=Reason.DAMAGED, at=NOW):
    return RefundRequest(rid, oid, amount, "USD", reason, at)


@pytest.fixture
def conn():
    c = connect()
    insert_order(c, make_order())
    insert_order(c, make_order("o_big", paid=50_000))
    insert_order(c, make_order("o_paid", status=OrderStatus.PAID))
    return c


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def service(conn, provider):
    return RefundService(conn, provider, now=NOW)
