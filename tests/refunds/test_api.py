import pytest
from fastapi.testclient import TestClient

from refunds.api import create_app


@pytest.fixture()
def client(service):
    return TestClient(create_app(service))


def post(client, **kw):
    body = {"request_id": "r_1", "order_id": "o_1", "amount": "40.00", "reason": "damaged"}
    body.update(kw)
    return client.post("/refund-requests", json=body)


def test_submit_then_retry_is_byte_identical_and_single_refund(client):
    a = post(client)
    b = post(client)
    assert a.status_code == 200 and a.json() == b.json()
    assert a.json()["decision"]["outcome"] == "approve" and a.json()["refund"]["status"] == "settled"
    ledger = client.get("/orders/o_1/refunds").json()
    assert len(ledger["refunds"]) == 1 and ledger["settled"] == "40.00" and ledger["remaining"] == "60.00"


def test_reused_request_id_with_new_payload_is_409(client):
    post(client)
    assert post(client, amount="41.00").status_code == 409


def test_unknown_order_404_and_bad_amount_422(client):
    assert post(client, order_id="ghost").status_code == 404
    assert post(client, amount="abc").status_code == 422
    assert post(client, amount=12.5).status_code == 422  # floats are not accepted


def test_escalation_returns_reasons_and_no_refund(client):
    r = post(client, reason="other")
    assert r.json()["decision"]["outcome"] == "escalate" and r.json()["refund"] is None
    assert r.json()["decision"]["reasons"] == ["reason_requires_review:other"]
