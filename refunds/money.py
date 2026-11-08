"""Exact money: Decimal quantised to cents, currency-checked. Floats are rejected on purpose."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import Union

CENT = Decimal("0.01")
Amountish = Union[str, int, Decimal]


class CurrencyMismatch(ValueError):
    pass


@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str = "USD"

    def __post_init__(self) -> None:
        if isinstance(self.amount, float):
            raise TypeError("Money does not accept float; pass str or Decimal")
        try:
            q = Decimal(self.amount).quantize(CENT, rounding=ROUND_HALF_EVEN)
        except InvalidOperation as e:  # e.g. Decimal("abc")
            raise ValueError(f"invalid money amount {self.amount!r}") from e
        object.__setattr__(self, "amount", q)
        object.__setattr__(self, "currency", self.currency.upper())

    @classmethod
    def of(cls, value: Amountish, currency: str = "USD") -> "Money":
        if isinstance(value, float):
            raise TypeError("Money does not accept float; pass str or Decimal")
        return cls(Decimal(value), currency)

    @classmethod
    def zero(cls, currency: str = "USD") -> "Money":
        return cls(Decimal(0), currency)

    def _check(self, other: "Money") -> None:
        if not isinstance(other, Money):
            raise TypeError(f"cannot combine Money with {type(other).__name__}")
        if other.currency != self.currency:
            raise CurrencyMismatch(f"{self.currency} vs {other.currency}")

    def __add__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.amount - other.amount, self.currency)

    def __lt__(self, other: "Money") -> bool:
        self._check(other)
        return self.amount < other.amount

    def __le__(self, other: "Money") -> bool:
        self._check(other)
        return self.amount <= other.amount

    def __gt__(self, other: "Money") -> bool:
        self._check(other)
        return self.amount > other.amount

    def __ge__(self, other: "Money") -> bool:
        self._check(other)
        return self.amount >= other.amount

    def min(self, other: "Money") -> "Money":
        self._check(other)
        return self if self.amount <= other.amount else other

    def scale(self, factor: Amountish) -> "Money":
        """Multiply by a Decimal factor (used for policy ceilings), rounding half-even to cents."""
        return Money(self.amount * Decimal(factor), self.currency)

    @property
    def is_positive(self) -> bool:
        return self.amount > 0

    @property
    def is_zero(self) -> bool:
        return self.amount == 0

    def __str__(self) -> str:
        return f"{self.amount:.2f} {self.currency}"
