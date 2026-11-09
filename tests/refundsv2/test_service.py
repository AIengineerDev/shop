import pytest

from refundsv2.ledger import UnknownOrder, ledger_view, list_refunds
from refundsv2.model import Outcome, Reason, RefundStatus
from refundsv2.policy import PolicyConfig
from refundsv2.provider import FakeProvider
from refundsv2.service import RefundService
from tests.refundsv2.conftest import NOW, req


def test_approve_settles_and_holds_money(service, provider, conn):
    r = service.handle(req(), NOW)
    assert r.status == RefundStatus.SETTLED and r.amount_cents == 4_000 and r.provider_ref == "pr_r1"
    assert r.decision.outcome == Outcome.APPROVE and r.decision.policy_version == "v1"
    assert ledger_view(conn, "o1").remaining_cents == 6_000
    assert provider.calls == [("r1", 4_000, "USD")]


def test_hold_is_written_before_the_provider_is_called(conn):
    class Crashing(FakeProvider):
        def refund(self, *a):
            raise RuntimeError("process died mid-call")
    svc = RefundService(conn, Crashing(), now=NOW)
    with pytest.raises(RuntimeError):
        svc.handle(req(), NOW)
    rows = list_refunds(conn, "o1")
    assert len(rows) == 1 and rows[0].status == RefundStatus.PENDING
    assert ledger_view(conn, "o1").held_cents == 4_000  # money held, never free


def test_declined_releases_the_hold(service, provider, conn):
    provider.script["r1"] = ["declined"]
    r = service.handle(req(), NOW)
    assert r.status == RefundStatus.FAILED and r.attempts == 1
    assert ledger_view(conn, "o1").remaining_cents == 10_000


def test_escalate_and_reject_are_recorded_without_holding(service, provider, conn):
    e = service.handle(req("e", reason=Reason.OTHER), NOW)
    assert e.status == RefundStatus.ESCALATED and e.decision.outcome == Outcome.ESCALATE and e.amount_cents == 0
    x = service.handle(req("x", amount=0), NOW)
    assert x.status == RefundStatus.REJECTED and x.decision.reasons == ("amount_invalid:must_be_positive",)
    assert ledger_view(conn, "o1").held_cents == 0 and provider.calls == []


def test_unknown_order(service):
    with pytest.raises(UnknownOrder):
        service.handle(req(oid="nope"), NOW)


def test_partials_to_the_cent_then_capped_then_rejected(service, conn):
    service.set_policy(PolicyConfig(version="lenient", max_refunds_90d=99), NOW)  # history rule is tested separately
    for i in range(3):
        assert service.handle(req(f"p{i}", 3_333), NOW).status == RefundStatus.SETTLED
    capped = service.handle(req("p3", 500), NOW)
    assert capped.status == RefundStatus.SETTLED and capped.amount_cents == 1 and "capped_to_remaining:0.01" in capped.decision.reasons
    last = service.handle(req("p4", 1), NOW)
    assert last.status == RefundStatus.REJECTED and last.decision.reasons == ("nothing_left_to_refund",)
    assert ledger_view(conn, "o1").remaining_cents == 0


def test_history_signal_comes_from_the_ledger(service):
    for i in range(3):
        service.handle(req(f"h{i}", 100), NOW)
    d = service.handle(req("h3", 100), NOW)
    assert d.status == RefundStatus.ESCALATED and "history_threshold:3>=3" in d.decision.reasons


def test_abuse_score_flows_through(service):
    assert service.handle(req("a1", 100), NOW, abuse_score=0.95).status == RefundStatus.ESCALATED
    assert service.handle(req("a2", 100), NOW, abuse_score=0.1).status == RefundStatus.SETTLED
