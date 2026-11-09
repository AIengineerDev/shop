"""Money as integer minor units (cents).

Why integers: every partial refund sums exactly, storage is a plain INTEGER column, and SQLite can
enforce the ceiling in a WHERE clause without any decimal type. `Money` only formats and checks
currency; arithmetic on cents is ordinary integer arithmetic. Floats are rejected on input because
0.1 + 0.2 is not 0.3 and a refund system should never have to explain that to a customer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Union

_AMOUNT = re.compile(r"^\s*(\d+)(?:\.(\d{1,2}))?\s*$")


class InvalidAmount(ValueError):
    pass


def parse_amount(value: Union[str, int, Decimal]) -> int:
    """'40', '40.5', '40.05', 40, Decimal('40.05') -> cents. Floats, negatives, >2 decimals rejected."""
    if isinstance(value, bool) or isinstance(value, float):
        raise InvalidAmount(f"amount must be a string or integer, not {type(value).__name__}")
    if isinstance(value, int):
        if value < 0:
            raise InvalidAmount("amount must be >= 0")
        return value * 100
    text = str(value)
    m = _AMOUNT.match(text)
    if not m:
        raise InvalidAmount(f"amount {text!r} is not a non-negative decimal with at most 2 places")
    whole, frac = m.group(1), (m.group(2) or "").ljust(2, "0")
    return int(whole) * 100 + int(frac)


def format_cents(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{cents // 100}.{cents % 100:02d}"


class CurrencyMismatch(ValueError):
    pass


@dataclass(frozen=True)
class Money:
    cents: int
    currency: str = "USD"

    @classmethod
    def parse(cls, value: Union[str, int, Decimal], currency: str = "USD") -> "Money":
        return cls(parse_amount(value), currency)

    def _check(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise CurrencyMismatch(f"{self.currency} vs {other.currency}")

    def __add__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.cents + other.cents, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.cents - other.cents, self.currency)

    def __str__(self) -> str:
        return f"{format_cents(self.cents)} {self.currency}"
