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

    def lookup(self, idempotency_key: str) -> Optional[ProviderReceipt]:
        """Did a refund with this key execute? Used by reconciliation after a timeout."""
        ...


class FakeProvider:
    """mode: 'ok' | 'decline' | 'timeout'. Calls are recorded by idempotency key.

    In 'timeout' mode `executed_before_timeout` says whether the money actually moved before the
    connection dropped — the two cases reconciliation must tell apart.
    """

    def __init__(self, mode: str = "ok", executed_before_timeout: bool = False) -> None:
        self.mode = mode
        self.executed_before_timeout = executed_before_timeout
        self.calls: Dict[str, Money] = {}
        self._settled: Dict[str, ProviderReceipt] = {}

    def refund(self, idempotency_key: str, amount: Money) -> ProviderReceipt:
        self.calls[idempotency_key] = amount
        if idempotency_key in self._settled:  # provider-side idempotency
            return self._settled[idempotency_key]
        if self.mode == "decline":
            raise ProviderDeclined("insufficient balance on original charge")
        if self.mode == "timeout":
            if self.executed_before_timeout:
                self._execute(idempotency_key, amount)
            raise ProviderTimeout("gateway timeout")
        return self._execute(idempotency_key, amount)

    def lookup(self, idempotency_key: str) -> Optional[ProviderReceipt]:
        return self._settled.get(idempotency_key)

    def _execute(self, key: str, amount: Money) -> ProviderReceipt:
        receipt = ProviderReceipt(f"prov_{len(self._settled) + 1:06d}", amount)
        self._settled[key] = receipt
        return receipt
