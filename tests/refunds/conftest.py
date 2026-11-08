from datetime import datetime, timedelta, timezone

import pytest

from refunds.config import PolicyConfig
from refunds.ledger import RefundLedger
from refunds.model import LedgerView, Order, OrderStatus, RefundReason, RefundRequest
from refunds.money import Money
from refunds.provider import FakeProvider
from refunds.service import OrderRepo, RefundService

NOW = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
DAY = timedelta(days=1)


def order(order_id="o_1", customer="c_1", paid="100.00", status=OrderStatus.DELIVERED,
          paid_at=NOW - 5 * DAY, delivered_at=NOW - 2 * DAY):
    return Order(order_id, customer, Money.of(paid), status, paid_at,
                 delivered_at if status is OrderStatus.DELIVERED else None)


def request(request_id="r_1", order_id="o_1", amount="40.00", reason=RefundReason.DAMAGED, at=NOW):
    return RefundRequest(request_id, order_id, Money.of(amount), reason, at)


def view(paid="100.00", settled="0.00", reserved="0.00"):
    return LedgerView(Money.of(paid), Money.of(settled), Money.of(reserved))


@pytest.fixture()
def policy():
    return PolicyConfig()


@pytest.fixture()
def provider():
    return FakeProvider("ok")


def make_service(orders, provider=None, policy=None, clock=lambda: NOW, **kw):
    return RefundService(OrderRepo(orders), RefundLedger(), provider or FakeProvider("ok"),
                         policy or PolicyConfig(), clock=clock, **kw)


@pytest.fixture()
def service(provider):
    return make_service([order(), order("o_2", paid="500.00"), order("o_3", status=OrderStatus.PAID, delivered_at=None)],
                        provider=provider)
