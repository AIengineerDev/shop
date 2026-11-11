from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from tests.conftest import FROZEN_NOW  # noqa: F401  (import keeps seed/now consistent)


def _walk(obj: Any, path: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_walk(v, f"{path}.{k}" if path else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_walk(v, f"{path}[{i}]"))
    else:
        out.append((path, obj))
    return out


def assert_money_strings(body: Any) -> None:
    """AC-A3: any key that denotes money is a 2-decimal string or null, never a number."""
    for path, val in _walk(body):
        key = path.rsplit(".", 1)[-1].split("[")[0]
        if key.endswith("price") or key in {
            "delta",
            "coupon",
            "shipping",
            "new_price",
            "old_price",
        }:
            assert val is None or (isinstance(val, str) and len(val.split(".")[-1]) == 2), (
                path,
                val,
            )


def test_ac_a1_full_flow(api: TestClient) -> None:
    r = api.post("/v1/runs", json={"stages": ["extract", "recommend"]})
    assert r.status_code == 202, r.text
    run = r.json()
    assert run["status"] == "completed" and run["skus_considered"] == 50
    assert run["offers_extracted"] == 200

    r = api.get("/v1/recommendations", params={"status": "pending", "decision": "propose"})
    assert r.status_code == 200
    page = r.json()
    assert page["total"] >= 1
    assert_money_strings(page)

    # worked example SKU-0001 must be the spec's AC-R2 numbers
    r = api.get("/v1/recommendations", params={"sku": "SKU-0001", "status": "pending"})
    rec = r.json()["items"][0]
    assert rec["decision"] == "propose"
    assert rec["current_price"] == "45.00"
    assert rec["floor_price"] == "36.00"
    assert rec["lowest_competitor_price"] == "37.99"
    assert rec["proposed_price"] == "40.99"
    # The scrape text does not mention shipping, so SHIPPING_UNKNOWN is part of the derivation.
    assert rec["reason_codes"] == [
        "UNDERCUT",
        "SHIPPING_UNKNOWN",
        "CLAMP_MAX_DECREASE",
        "ROUNDED",
    ]

    card = api.get(f"/v1/recommendations/{rec['recommendation_id']}").json()
    assert card["title"].endswith("Headphones X1")
    assert len(card["eligible_offers"]) == 2
    assert any("$42.99 with $5 coupon" in o["raw_text"] for o in card["eligible_offers"])
    assert "40.99" in card["explanation"]
    assert_money_strings(card)

    r = api.post(
        f"/v1/recommendations/{rec['recommendation_id']}/approve",
        json={"actor": "alex", "note": "ok"},
    )
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["review_status"] == "applied" and res["new_price"] == "40.99"
    assert res["change_id"]

    hist = api.get("/v1/skus/SKU-0001/history").json()
    assert len(hist) == 1
    assert hist[0]["old_price"] == "45.00" and hist[0]["new_price"] == "40.99"
    assert hist[0]["source"] == "review_approve" and hist[0]["action"] == "approve"
    assert_money_strings(hist)

    v = api.get("/v1/audit/verify")
    assert v.status_code == 200 and v.json() == {"ok": True, "breaks": []}

    # second approve -> 409 envelope
    r = api.post(f"/v1/recommendations/{rec['recommendation_id']}/approve", json={"actor": "alex"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "ILLEGAL_TRANSITION"

    offers = api.get("/v1/skus/SKU-0001/offers").json()
    assert len(offers) == 2 and all(o["status"] == "accepted" for o in offers)
    assert_money_strings(offers)

    # new run expires the leftover pendings for other SKUs and makes new ones
    r2 = api.post("/v1/runs", json={"stages": ["recommend"]}).json()
    assert r2["offers_extracted"] == 0 and r2["recommendations"] == 50
    run_report = api.get(f"/v1/runs/{r2['run_id']}").json()
    assert run_report["status"] == "completed"


def test_ac_a2_override(api: TestClient) -> None:
    api.post("/v1/runs", json={})
    rec = api.get("/v1/recommendations", params={"sku": "SKU-0001", "status": "pending"}).json()[
        "items"
    ][0]
    rid = rec["recommendation_id"]
    r = api.post(f"/v1/recommendations/{rid}/override", json={"actor": "a", "price": "35.99"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "FLOOR_VIOLATION"
    r = api.post(f"/v1/recommendations/{rid}/override", json={"actor": "a", "price": "41.999"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "MONEY_INVALID"
    r = api.post(f"/v1/recommendations/{rid}/override", json={"actor": "a", "price": "41.99"})
    assert r.status_code == 200, r.text
    assert r.json()["new_price"] == "41.99"
    hist = api.get("/v1/skus/SKU-0001/history").json()
    assert hist[-1]["source"] == "override" and hist[-1]["new_price"] == "41.99"
    card = api.get(f"/v1/recommendations/{rid}").json()
    assert card["actions"][0]["action"] == "override"
    assert card["actions"][0]["override_price"] == "41.99"


def test_ac_a3_no_numeric_money_anywhere(api: TestClient) -> None:
    api.post("/v1/runs", json={})
    for path in ("/v1/recommendations?limit=500", "/v1/skus/SKU-0002/offers", "/v1/policies"):
        body = api.get(path).json()
        assert_money_strings(body)
    for item in api.get("/v1/recommendations?limit=500").json()["items"][:10]:
        assert_money_strings(api.get(f"/v1/recommendations/{item['recommendation_id']}").json())
    raw = api.get("/v1/recommendations?limit=5").text
    # belt and braces: no bare JSON number directly after a *_price key
    assert '"price": ' not in raw.replace('"price": "', "")


def test_reject_and_hold_rules(api: TestClient) -> None:
    api.post("/v1/runs", json={})
    hold = api.get("/v1/recommendations", params={"decision": "hold", "status": "pending"}).json()
    assert hold["total"] >= 1
    hid = hold["items"][0]["recommendation_id"]
    r = api.post(f"/v1/recommendations/{hid}/approve", json={"actor": "a"})
    assert r.status_code == 409
    r = api.post(f"/v1/recommendations/{hid}/reject", json={"actor": "a", "note": "n"})
    assert r.status_code == 200 and r.json()["review_status"] == "rejected"
    assert api.get("/v1/recommendations/nope").status_code == 404
    assert api.get("/v1/skus/NOPE/history").status_code == 404
    assert api.get("/v1/runs/nope").status_code == 404
    bad = api.post("/v1/recommendations/x/approve", json={"actor": ""})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "VALIDATION_ERROR"


def test_ac_a4_policy_reload_rollback(api: TestClient) -> None:
    before = api.get("/v1/policies").json()
    assert before["version"] == 1 and "default" in before["policies"]
    path = api.app.state.settings.policies_path  # type: ignore[attr-defined]
    good = path.read_text()
    path.write_text("default:\n  strategy: nonsense\n")
    r = api.post("/v1/policies/reload")
    assert r.status_code == 422 and r.json()["error"]["code"] == "POLICY_INVALID"
    after = api.get("/v1/policies").json()
    assert after == before
    path.write_text(good.replace("undercut_cents: 50", "undercut_cents: 75"))
    r = api.post("/v1/policies/reload")
    assert r.status_code == 200 and r.json()["version"] == 2
    assert r.json()["policies"]["default"]["undercut_cents"] == 75
    assert api.get("/healthz").json()["status"] == "ok"


def test_tampered_chain_reported(api: TestClient) -> None:
    import sqlite3

    api.post("/v1/runs", json={})
    rec = api.get("/v1/recommendations", params={"sku": "SKU-0001", "status": "pending"}).json()[
        "items"
    ][0]
    api.post(f"/v1/recommendations/{rec['recommendation_id']}/approve", json={"actor": "a"})
    db = api.app.state.settings.db_path  # type: ignore[attr-defined]
    conn = sqlite3.connect(db)
    conn.execute("DROP TRIGGER trg_price_changes_no_update")
    conn.execute("UPDATE price_changes SET actor = 'mallory'")
    conn.commit()
    conn.close()
    v = api.get("/v1/audit/verify")
    assert v.status_code == 409 and v.json()["ok"] is False
    assert v.json()["breaks"][0]["sku"] == "SKU-0001"
    _ = json
