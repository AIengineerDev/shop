"""Money primitives. Integer cents and basis points everywhere; Decimal only at the edges.

Rule: nothing in this module or in ``repricer.engine`` may use ``float``. A test walks the
AST to enforce it.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import NewType

Cents = NewType("Cents", int)
Bp = NewType("Bp", int)

BP_SCALE = 10_000
CENT = Decimal("0.01")
ZERO = Cents(0)


class MoneyError(ValueError):
    pass


def to_cents(value: Decimal | str | int) -> Cents:
    """Convert a decimal dollar amount to cents. Rejects more than two decimal places."""
    try:
        d = Decimal(value)
    except InvalidOperation as e:  # pragma: no cover - defensive
        raise MoneyError(f"not a decimal: {value!r}") from e
    if not d.is_finite():
        raise MoneyError(f"non-finite amount: {value!r}")
    quantized = d.quantize(CENT, rounding=ROUND_HALF_EVEN)
    if quantized != d:
        raise MoneyError(f"amount has more than 2 decimal places: {value!r}")
    return Cents(int(quantized * 100))


def to_cents_rounded(value: Decimal) -> Cents:
    """Convert with banker's rounding to the cent. Used for computed intermediates only."""
    if not value.is_finite():
        raise MoneyError(f"non-finite amount: {value!r}")
    return Cents(int(value.quantize(CENT, rounding=ROUND_HALF_EVEN) * 100))


def from_cents(cents: int) -> Decimal:
    return (Decimal(cents) / 100).quantize(CENT)


def fmt(cents: int) -> str:
    """Canonical API representation: a string with exactly two decimals, e.g. "42.99"."""
    return f"{from_cents(cents):.2f}"


def to_bp(percent: Decimal | str | int) -> Bp:
    """15.5 (%) -> 1550 bp. Rejects more than 2 decimal places of percent."""
    d = Decimal(percent)
    scaled = d * 100
    if scaled != scaled.to_integral_value():
        raise MoneyError(f"percent has more than 2 decimal places: {percent!r}")
    return Bp(int(scaled))


def bp_to_percent(bp: int) -> Decimal:
    return (Decimal(bp) / 100).quantize(CENT)


def apply_bp(cents: int, bp: int) -> Cents:
    """cents * (1 + bp/10000), banker's rounded to the cent."""
    raw = Decimal(cents) * (BP_SCALE + bp) / BP_SCALE
    return Cents(int(raw.quantize(Decimal(1), rounding=ROUND_HALF_EVEN)))


def ceil_cents(cents: int, bp: int) -> Cents:
    """ceil(cents * (1 + bp/10000)). Used for margin floors: never round a floor down."""
    raw = Decimal(cents) * (BP_SCALE + bp) / BP_SCALE
    return Cents(int(raw.quantize(Decimal(1), rounding=ROUND_CEILING)))


def fraction_bp(cents: int, bp: int) -> Cents:
    """cents * bp/10000, banker's rounded. e.g. 10% of $42.99 -> 430 cents."""
    raw = Decimal(cents) * bp / BP_SCALE
    return Cents(int(raw.quantize(Decimal(1), rounding=ROUND_HALF_EVEN)))


def delta_bp(old: int, new: int) -> Bp:
    """(new - old) / old in basis points, banker's rounded. old must be > 0."""
    if old <= 0:
        raise MoneyError("delta_bp requires old > 0")
    raw = Decimal(new - old) * BP_SCALE / old
    return Bp(int(raw.quantize(Decimal(1), rounding=ROUND_HALF_EVEN)))


def round_to(cents: int, ending: int, floor: int) -> Cents:
    """Round so the price ends in ``ending`` cents (e.g. 99 -> $x.99).

    Picks the nearest candidate (ties go down). If the nearest candidate is below ``floor``,
    rounds up to the next candidate instead. ``ending=0`` disables rounding (but still
    enforces the floor).
    """
    if not 0 <= ending <= 99:
        raise MoneyError(f"ending must be 0..99, got {ending}")
    if ending == 0:
        return Cents(max(cents, floor))
    dollars = cents // 100
    below = dollars * 100 + ending
    if below > cents:
        below -= 100
    above = below + 100
    nearest = below if (cents - below) <= (above - cents) else above
    if nearest < floor:
        candidate = above
        while candidate < floor:
            candidate += 100
        return Cents(candidate)
    return Cents(nearest)


def clamp(value: int, lo: int, hi: int) -> Cents:
    if lo > hi:
        raise MoneyError(f"clamp bounds inverted: lo={lo} hi={hi}")
    return Cents(min(max(value, lo), hi))
