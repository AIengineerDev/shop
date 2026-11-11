from decimal import Decimal

import pytest

from repricer import money
from repricer.money import MoneyError


def test_to_cents_exact() -> None:
    assert money.to_cents("42.99") == 4299
    assert money.to_cents(Decimal("0.01")) == 1
    assert money.to_cents(5) == 500


def test_to_cents_rejects_sub_cent() -> None:
    with pytest.raises(MoneyError):
        money.to_cents("42.999")


def test_fmt_two_decimals() -> None:
    assert money.fmt(4299) == "42.99"
    assert money.fmt(5) == "0.05"
    assert money.fmt(100000) == "1000.00"


def test_to_bp() -> None:
    assert money.to_bp("20") == 2000
    assert money.to_bp("12.5") == 1250
    with pytest.raises(MoneyError):
        money.to_bp("12.345")


def test_ceil_cents_never_rounds_down() -> None:
    # 30.00 * 1.20 = 36.00 exactly
    assert money.ceil_cents(3000, 2000) == 3600
    # 10.01 * 1.15 = 11.5115 -> ceil to 11.52
    assert money.ceil_cents(1001, 1500) == 1152
    assert money.ceil_cents(1001, 0) == 1001


def test_apply_bp_bankers_rounding() -> None:
    assert money.apply_bp(4500, -1000) == 4050  # -10%
    assert money.apply_bp(4500, 500) == 4725  # +5%


def test_fraction_bp() -> None:
    assert money.fraction_bp(4299, 1000) == 430  # 10% of 42.99 = 4.299 -> 4.30


def test_delta_bp() -> None:
    assert money.delta_bp(4500, 4099) == -891
    assert money.delta_bp(100, 150) == 5000
    with pytest.raises(MoneyError):
        money.delta_bp(0, 1)


@pytest.mark.parametrize(
    "cents,ending,floor,expected",
    [
        (4050, 99, 0, 4099),  # nearest .99 is 40.99 (49 up vs 51 down)
        (4049, 99, 0, 3999),  # nearest is 39.99
        (4049, 99, 4000, 4099),  # floor pushes up
        (4099, 99, 0, 4099),  # already there
        (4000, 0, 0, 4000),  # no rounding
        (3500, 0, 3600, 3600),  # no rounding but floor enforced
        (10, 99, 0, 99),  # below first candidate -> 0.99 vs -0.01: nearest is 0.99
        (4049, 95, 4096, 4195),  # floor above the next candidate too: keep climbing
    ],
)
def test_round_to(cents: int, ending: int, floor: int, expected: int) -> None:
    assert money.round_to(cents, ending, floor) == expected


def test_round_to_rejects_bad_ending() -> None:
    with pytest.raises(MoneyError):
        money.round_to(100, 100, 0)


def test_clamp() -> None:
    assert money.clamp(5, 1, 10) == 5
    assert money.clamp(-5, 1, 10) == 1
    assert money.clamp(50, 1, 10) == 10
    with pytest.raises(MoneyError):
        money.clamp(5, 10, 1)
