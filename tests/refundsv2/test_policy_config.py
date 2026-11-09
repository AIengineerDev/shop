import pytest

from refundsv2.db import connect
from refundsv2.model import RefundStatus
from refundsv2.policy import DEFAULT_POLICY, PolicyConfig
from refundsv2.provider import FakeProvider
from refundsv2.service import PolicyVersionConflict, RefundService
from tests.refundsv2.conftest import NOW, req


def test_threshold_change_without_deploy_and_version_recorded(service):
    a = service.handle(req("a", 15_000, oid="o_big"), NOW)
    assert a.status == RefundStatus.SETTLED and a.decision.policy_version == "v1"
    service.set_policy(PolicyConfig(version="v2", auto_limit_cents=10_000), NOW)
    b = service.handle(req("b", 15_000, oid="o_big"), NOW)
    assert b.status == RefundStatus.ESCALATED and b.decision.policy_version == "v2"


def test_same_version_different_content_is_refused(service):
    with pytest.raises(PolicyVersionConflict):
        service.set_policy(PolicyConfig(version="v1", window_days=1), NOW)
    assert service.policy == DEFAULT_POLICY
    service.set_policy(DEFAULT_POLICY, NOW)  # identical content re-activates without error


def test_other_instances_pick_up_the_active_policy(conn, provider):
    a = RefundService(conn, provider, now=NOW)
    b = RefundService(conn, provider, now=NOW)
    a.set_policy(PolicyConfig(version="v3", window_days=1), NOW)
    assert b.policy.version == "v1"
    assert b.refresh_policy().version == "v3"
    fresh = RefundService(conn, provider, now=NOW)
    assert fresh.policy.version == "v3"


def test_policy_validation():
    with pytest.raises(Exception):
        PolicyConfig(version="x", window_days=-1)
    with pytest.raises(Exception):
        PolicyConfig(version="x", abuse_escalate_at=1.5)
