"""Reconciliation of UNKNOWN refunds (provider call timed out after we decided to pay).

While a refund is UNKNOWN the ledger keeps the money reserved, so nothing else can be paid out of
it. A periodic job asks the provider what happened, by *our* request_id, which we sent as the
provider's idempotency key:

  * provider has it           -> the money moved: SETTLED with the provider's reference
  * provider has nothing      -> safe to call again with the SAME key; provider-side idempotency
                                 makes a late duplicate impossible. Declined -> FAILED (released);
                                 timed out again -> stays UNKNOWN for the next run.

Releasing a hold on "not found" alone would be wrong: the original request could still be in
flight at the provider.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from refunds.ledger import RefundLedger
from refunds.model import RefundStatus
from refunds.provider import PaymentProvider, ProviderDeclined, ProviderTimeout


@dataclass
class ReconcileReport:
    settled: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    still_unknown: List[str] = field(default_factory=list)


def reconcile_unknown(ledger: RefundLedger, provider: PaymentProvider) -> ReconcileReport:
    rep = ReconcileReport()
    for r in ledger.with_status(RefundStatus.UNKNOWN):
        receipt = provider.lookup(r.request_id)
        if receipt is None:
            try:
                receipt = provider.refund(r.request_id, r.amount)
            except ProviderDeclined:
                ledger.fail(r.refund_id)
                rep.failed.append(r.refund_id)
                continue
            except ProviderTimeout:
                rep.still_unknown.append(r.refund_id)
                continue
        ledger.settle(r.refund_id, receipt.provider_ref)
        rep.settled.append(r.refund_id)
    return rep
