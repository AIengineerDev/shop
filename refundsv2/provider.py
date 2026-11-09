"""Payment provider boundary.

Two calls are all the system needs: `refund` (idempotent on our request_id, as every serious
provider supports) and `lookup` (what the provider believes happened to that key). The fake is
scripted per request_id so tests can produce declines, timeouts where the money did not move, and
timeouts where it did.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol


class ProviderTimeout(Exception):
    """The call did not return. The money may or may not have moved."""


@dataclass(frozen=True)
class ProviderResult:
    status: str                    # "ok" | "declined"
    ref: Optional[str] = None


class PaymentProvider(Protocol):
    def refund(self, request_id: str, amount_cents: int, currency: str) -> ProviderResult: ...
    def lookup(self, request_id: str) -> Optional[ProviderResult]: ...


@dataclass
class FakeProvider:
    """script[request_id] = sequence of outcomes for successive calls: ok | declined | timeout | timeout_moved.
    Unscripted calls succeed. `state` is what the provider actually did; `lookup` reads it."""
    script: Dict[str, List[str]] = field(default_factory=dict)
    calls: List[tuple] = field(default_factory=list)
    state: Dict[str, ProviderResult] = field(default_factory=dict)

    def refund(self, request_id: str, amount_cents: int, currency: str) -> ProviderResult:
        self.calls.append((request_id, amount_cents, currency))
        if request_id in self.state and self.state[request_id].status == "ok":
            return self.state[request_id]  # provider-side idempotency: same key, same answer, no second payout
        outcome = self.script.get(request_id, ["ok"])
        step = outcome.pop(0) if len(outcome) > 1 else outcome[0]
        if step == "ok":
            self.state[request_id] = ProviderResult("ok", f"pr_{request_id}")
            return self.state[request_id]
        if step == "declined":
            self.state[request_id] = ProviderResult("declined", None)
            return self.state[request_id]
        if step == "timeout_moved":
            self.state[request_id] = ProviderResult("ok", f"pr_{request_id}")
            raise ProviderTimeout(request_id)
        if step == "timeout":
            raise ProviderTimeout(request_id)
        raise ValueError(step)

    def lookup(self, request_id: str) -> Optional[ProviderResult]:
        return self.state.get(request_id)

    def calls_for(self, request_id: str) -> int:
        return sum(1 for c in self.calls if c[0] == request_id)
