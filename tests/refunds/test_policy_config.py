import pytest
from fastapi.testclient import TestClient

from refunds.api import create_app
from refunds.config import PolicyConfig
from refunds.model import Outcome
from tests.refunds.conftest import request


def test_config_from_dict_and_frozen():
    p = PolicyConfig.from_dict({"version": "v2", "auto_limit": "50.00", "window_days": 14})
    assert str(p.auto_limit) == "50.00" and p.window_days == 14
    with pytest.raises(Exception):
        p.window_days = 1  # type: ignore[misc]


def test_policy_swap_changes_outcome_and_both_records_carry_versions(service):
    a = service.handle(request("a", "o_2", "150.00"))
    assert a.decision.approved and a.refund.policy_version == "v1"
    service.set_policy(PolicyConfig(version="v2", auto_limit="100.00"))
    b = service.handle(request("b", "o_2", "150.00"))
    assert b.decision.outcome is Outcome.ESCALATE and b.decision.policy_version == "v2"
    assert "above_auto_limit:100.00 USD" in b.decision.reasons
    assert [r.policy_version for r in service.ledger.refunds_for("o_2")] == ["v1"]


def test_retry_after_policy_change_replays_original_decision(service):
    first = service.handle(request("a", "o_2", "150.00"))
    service.set_policy(PolicyConfig(version="v2", auto_limit="100.00"))
    assert service.handle(request("a", "o_2", "150.00")) is first


def test_same_version_different_content_is_refused(service):
    with pytest.raises(ValueError):
        service.set_policy(PolicyConfig(version="v1", auto_limit="1.00"))


def test_admin_endpoints(service):
    c = TestClient(create_app(service))
    assert c.get("/admin/policy").json()["active"]["version"] == "v1"
    r = c.put("/admin/policy", json={"version": "v2", "auto_limit": "20.00"})
    assert r.status_code == 200 and r.json()["auto_limit"] == "20.00"
    assert c.get("/admin/policy").json()["history"] == ["v1", "v2"]
    assert c.get("/health").json()["policy_version"] == "v2"
    assert c.put("/admin/policy", json={"version": "v3", "window_days": -1}).status_code == 422
    body = {"request_id": "x", "order_id": "o_1", "amount": "40.00", "reason": "damaged"}
    out = c.post("/refund-requests", json=body).json()["decision"]
    assert out["outcome"] == "escalate" and out["policy_version"] == "v2"
