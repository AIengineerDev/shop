"""Payment provider port + a fake with scripted behaviour for tests and demos."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

from refunds.money import Money


class ProviderDeclined(Exception):
    """The provider answered and said no. Money did not move."""


class ProviderTimeout(Exception):
    """No answer. Money may or may not have moved."""


@dataclass(frozen=True)
class ProviderReceipt:
    provider_ref: str
    amount: Money


class PaymentProvider(Protocol):
    def refund(self, idempotency_key: str, amount: Money) -> ProviderReceipt: ...


class FakeProvider:
    """mode: 'ok' | 'decline' | 'timeout'. Calls are recorded by idempotency key."""

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.calls: Dict[str, Money] = {}
        self._settled: Dict[str, ProviderReceipt] = {}

    def refund(self, idempotency_key: str, amount: Money) -> ProviderReceipt:
        self.calls[idempotency_key] = amount
        if idempotency_key in self._settled:  # provider-side idempotency
            return self._settled[idempotency_key]
        if self.mode == "decline":
            raise ProviderDeclined("insufficient balance on original charge")
        if self.mode == "timeout":
            raise ProviderTimeout("gateway timeout")
        receipt = ProviderReceipt(f"prov_{len(self._settled) + 1:06d}", amount)
        self._settled[idempotency_key] = receipt
        return receipt
