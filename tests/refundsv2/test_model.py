from refundsv2.model import Decision, LedgerView, Outcome, Reason, RefundRequest, RuleResult


def _req(**over):
    base = dict(request_id="r1", order_id="o1", amount_cents=4000, currency="USD", reason=Reason.DAMAGED, submitted_at=100.0)
    base.update(over)
    return RefundRequest(**base)


def test_fingerprint_ignores_timestamp_and_request_id_but_not_payload():
    assert _req().fingerprint() == _req(submitted_at=999.0, request_id="r2").fingerprint()
    assert _req().fingerprint() != _req(amount_cents=4001).fingerprint()
    assert _req().fingerprint() != _req(reason=Reason.OTHER).fingerprint()


def test_decision_json_round_trip():
    d = Decision(Outcome.ESCALATE, 0, ("outside_window:30d",), (RuleResult("within_window", False, "outside_window:30d", "escalate"),), "v1")
    assert Decision.from_json(d.to_json()) == d


def test_ledger_view_remaining():
    assert LedgerView(paid_cents=10000, held_cents=3333).remaining_cents == 6667
