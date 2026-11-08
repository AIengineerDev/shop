from decimal import Decimal

import pytest

from refunds.money import CurrencyMismatch, Money


def test_three_partials_sum_exactly():
    parts = [Money.of("33.33")] * 3
    total = sum(parts, Money.zero())
    assert total == Money.of("99.99")
    assert total < Money.of("100.00")
    assert Money.of("100.00") - total == Money.of("0.01")


def test_floats_are_rejected():
    with pytest.raises(TypeError):
        Money.of(0.1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Money(0.1)  # type: ignore[arg-type]


def test_quantises_half_even_to_cents():
    assert Money.of("1.005").amount == Decimal("1.00")
    assert Money.of("1.015").amount == Decimal("1.02")
    assert Money.of("200").scale("0.5") == Money.of("100.00")


def test_currency_mismatch():
    with pytest.raises(CurrencyMismatch):
        Money.of("1", "USD") + Money.of("1", "EUR")
    with pytest.raises(CurrencyMismatch):
        Money.of("1", "USD") < Money.of("1", "EUR")


def test_currency_normalised_and_str():
    m = Money.of("5", "usd")
    assert m.currency == "USD" and str(m) == "5.00 USD"
