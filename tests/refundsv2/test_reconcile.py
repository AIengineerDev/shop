from refundsv2.ledger import get_refund, ledger_view, transition
from refundsv2.model import RefundStatus
from tests.refundsv2.conftest import NOW, req


def test_timeout_leaves_money_held_as_unknown(service, provider, conn):
    provider.script["r1"] = ["timeout"]
    r = service.handle(req(), NOW)
    assert r.status == RefundStatus.UNKNOWN and r.attempts == 1
    assert ledger_view(conn, "o1").held_cents == 4_000
    assert service.handle(req("r2", 7_000), NOW).amount_cents == 6_000  # capped around the unknown hold


def test_lookup_finds_money_moved_then_settles(service, provider, conn):
    provider.script["r1"] = ["timeout_moved"]
    service.handle(req(), NOW)
    assert service.reconcile(NOW + 1) == [("r1", "settled")]
    assert get_refund(conn, "r1").status == RefundStatus.SETTLED and provider.calls_for("r1") == 1


def test_not_found_waits_for_grace_then_reissues_under_same_key(service, provider, conn):
    provider.script["r1"] = ["timeout", "ok"]
    service.handle(req(), NOW)
    assert service.reconcile(NOW + 10) == [("r1", "waiting")]
    assert get_refund(conn, "r1").status == RefundStatus.UNKNOWN
    assert service.reconcile(NOW + 400) == [("r1", "reissued")]
    r = get_refund(conn, "r1")
    assert r.status == RefundStatus.SETTLED and r.attempts == 2 and provider.calls == [("r1", 4_000, "USD")] * 2


def test_reissue_is_idempotent_at_the_provider(service, provider, conn):
    """Provider actually paid, our record is UNKNOWN, lookup is somehow empty: re-issuing the same key cannot pay twice."""
    provider.script["r1"] = ["timeout_moved"]
    service.handle(req(), NOW)
    provider.state.pop("r1")                      # lookup now says not found
    provider.state["r1"] = provider.refund.__self__.state.get("r1") if False else None
    provider.state.pop("r1", None)
    provider.script["r1"] = ["ok"]
    service.reconcile(NOW + 400)
    assert get_refund(conn, "r1").status == RefundStatus.SETTLED and provider.calls_for("r1") == 2


def test_declined_on_lookup_releases(service, provider, conn):
    provider.script["r1"] = ["timeout"]
    service.handle(req(), NOW)
    provider.state["r1"] = provider.lookup("r1") or __import__("refundsv2.provider", fromlist=["ProviderResult"]).ProviderResult("declined")
    assert service.reconcile(NOW + 1) == [("r1", "failed")]
    assert ledger_view(conn, "o1").remaining_cents == 10_000


def test_stale_pending_is_treated_like_unknown(service, provider, conn):
    provider.script["r1"] = ["timeout_moved"]
    service.handle(req(), NOW)
    transition(conn, "r1", [RefundStatus.UNKNOWN], RefundStatus.PENDING, NOW)   # simulate a crash that left PENDING
    assert service.reconcile(NOW + 1) == []                                      # too young to touch
    assert service.reconcile(NOW + 400) == [("r1", "settled")]
