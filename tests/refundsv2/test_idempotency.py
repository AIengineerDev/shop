import pytest

from refundsv2.ledger import list_refunds
from refundsv2.model import Reason, RefundStatus
from refundsv2.policy import PolicyConfig
from refundsv2.service import DuplicateRequestConflict
from tests.refundsv2.conftest import NOW, req


def test_same_request_twice_pays_once(service, provider, conn):
    first = service.handle(req(), NOW)
    second = service.handle(req(at=NOW + 60), NOW + 60)   # retried a minute later
    assert first == second
    assert len(list_refunds(conn, "o1")) == 1 and provider.calls_for("r1") == 1


def test_same_id_different_payload_is_a_conflict(service):
    service.handle(req(), NOW)
    with pytest.raises(DuplicateRequestConflict):
        service.handle(req(amount=4_001), NOW)
    with pytest.raises(DuplicateRequestConflict):
        service.handle(req(reason=Reason.LATE), NOW)


def test_retry_of_escalated_and_rejected_replays(service):
    e1 = service.handle(req("e", reason=Reason.OTHER), NOW)
    assert e1.status == RefundStatus.ESCALATED and service.handle(req("e", reason=Reason.OTHER), NOW) == e1


def test_retry_after_policy_change_replays_original_decision(service):
    first = service.handle(req(), NOW)
    service.set_policy(PolicyConfig(version="v2", auto_limit_cents=100), NOW)
    assert service.handle(req(), NOW + 1) == first and first.decision.policy_version == "v1"
    fresh = service.handle(req("r2", 4_000), NOW + 1)
    assert fresh.status == RefundStatus.ESCALATED and fresh.decision.policy_version == "v2"


def test_retry_after_timeout_does_not_pay_twice(service, provider, conn):
    provider.script["r1"] = ["timeout_moved"]
    r1 = service.handle(req(), NOW)
    assert r1.status == RefundStatus.UNKNOWN
    r2 = service.handle(req(), NOW + 5)      # client retries the request itself
    assert r2 == r1 and provider.calls_for("r1") == 1
    service.reconcile(NOW + 10)
    assert service.handle(req(), NOW + 11).status == RefundStatus.SETTLED
    assert provider.calls_for("r1") == 1 and len(list_refunds(conn, "o1")) == 1
