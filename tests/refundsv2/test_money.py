import pytest
from decimal import Decimal

from refundsv2.money import CurrencyMismatch, InvalidAmount, Money, format_cents, parse_amount


@pytest.mark.parametrize("raw,cents", [("40", 4000), ("40.00", 4000), ("40.5", 4050), ("40.05", 4005), ("0", 0), ("0.01", 1), (40, 4000), (Decimal("33.33"), 3333)])
def test_parse(raw, cents):
    assert parse_amount(raw) == cents


@pytest.mark.parametrize("raw", ["40.005", "-1", "abc", "", "1e3", 40.0, True, "1,000"])
def test_parse_rejects(raw):
    with pytest.raises(InvalidAmount):
        parse_amount(raw)


def test_three_partials_sum_exactly():
    assert sum(parse_amount("33.33") for _ in range(3)) == parse_amount("99.99")
    assert format_cents(3 * 3333) == "99.99"


def test_format():
    assert format_cents(5) == "0.05" and format_cents(12345) == "123.45" and format_cents(-250) == "-2.50"


def test_money_arithmetic_and_currency_check():
    a, b = Money.parse("10.10"), Money.parse("0.90")
    assert (a + b).cents == 1100 and (a - b).cents == 920 and str(a) == "10.10 USD"
    with pytest.raises(CurrencyMismatch):
        a + Money(1, "EUR")
