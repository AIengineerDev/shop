import pytest
from fastapi.testclient import TestClient

from refundsv2.api import create_app
from tests.refundsv2.conftest import NOW


@pytest.fixture
def client(service):
    return TestClient(create_app(service, clock=lambda: NOW))


def body(**over):
    b = {"request_id": "r1", "order_id": "o1", "amount": "40.00", "reason": "damaged"}
    b.update(over)
    return b


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "policy_version": "v1"}


def test_submit_and_replay_identical(client, provider):
    a = client.post("/refund-requests", json=body())
    assert a.status_code == 200
    j = a.json()
    assert j["status"] == "settled" and j["amount"] == "40.00" and j["outcome"] == "approve" and j["policy_version"] == "v1"
    assert [t["rule"] for t in j["trace"]][:3] == ["order_refundable", "amount_valid", "cap_to_remaining"]
    b = client.post("/refund-requests", json=body())
    assert b.json() == j and provider.calls_for("r1") == 1


def test_conflict_on_reused_id(client):
    client.post("/refund-requests", json=body())
    assert client.post("/refund-requests", json=body(amount="41.00")).status_code == 409


@pytest.mark.parametrize("amount", [40.0, "40.005", "-1", "abc"])
def test_bad_amounts_are_422(client, amount):
    assert client.post("/refund-requests", json=body(amount=amount)).status_code == 422


def test_unknown_order_404_and_bad_reason_422(client):
    assert client.post("/refund-requests", json=body(order_id="nope")).status_code == 404
    assert client.post("/refund-requests", json=body(reason="because")).status_code == 422


def test_order_view_shows_ledger(client):
    client.post("/refund-requests", json=body())
    client.post("/refund-requests", json=body(request_id="r2", amount="70.00"))
    v = client.get("/orders/o1/refunds").json()
    assert v["paid"] == "100.00" and v["held"] == "100.00" and v["remaining"] == "0.00"
    assert [r["amount"] for r in v["refunds"]] == ["40.00", "60.00"]
    assert client.get("/orders/nope/refunds").status_code == 404


def test_admin_policy_and_reconcile(client, provider):
    r = client.put("/admin/policy", json={"version": "v2", "auto_limit_cents": 1000})
    assert r.status_code == 200 and client.get("/health").json()["policy_version"] == "v2"
    assert client.post("/refund-requests", json=body()).json()["status"] == "escalated"
    assert client.put("/admin/policy", json={"version": "v2", "auto_limit_cents": 5}).status_code == 409
    provider.script["t1"] = ["timeout_moved"]
    assert client.post("/refund-requests", json=body(request_id="t1", amount="5.00")).json()["status"] == "unknown"
    assert client.post("/admin/reconcile").json() == {"actions": [{"request_id": "t1", "action": "settled"}]}


def test_demo_app_boots():
    from refundsv2.api import app
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.post("/refund-requests", json=body(order_id="o_1")).json()["status"] == "settled"
