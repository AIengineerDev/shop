"""API tests + edge cases + business cases that need the full trained service."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from shopfeed.config import DAY, HEADS, Config, ValueWeights
from shopfeed.eval.ab import compare
from shopfeed.eval.offline import ips_estimate
from shopfeed.eval.simulator import simulate_responses
from shopfeed.ranking.value import value_score
from shopfeed.schema import FeedRequest
from shopfeed.serving.api import create_app
from tests.conftest import NOW, make_service


def _feed(client, user_id="u_0", country="US", **kw):
    return client.post("/feed", json={"user_id": user_id, "country": country, "session_id": "s1", "now": NOW, **kw})


@pytest.fixture()
def client(service):
    return TestClient(create_app(service))


def _user(world, cohort):
    return world["users"][world["users"].cohort == cohort].iloc[0]


# ---------------------------------------------------------------- API
def test_health(client, service):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_version"] == service.model_version


def test_feed_returns_page_with_required_fields(client, cfg, world):
    u = _user(world, "normal")
    r = _feed(client, u.user_id, u.country)
    assert r.status_code == 200
    body = r.json()
    assert len(body["cards"]) == cfg.rerank.page_size and not body["fallback"]
    card = body["cards"][0]
    for f in ["impression_id", "product_id", "merchant_id", "category", "price", "position", "source", "value", "is_exploration", "reason"]:
        assert f in card
    assert [c["position"] for c in body["cards"]] == list(range(cfg.rerank.page_size))
    assert body["timings_ms"]["total"] < cfg.latency_budget_ms * 5  # generous on CI boxes; eval script reports real p95


def test_unknown_user_gets_cold_start_page_not_500(client, cfg):
    r = _feed(client, "never_seen_before", "US")
    assert r.status_code == 200 and len(r.json()["cards"]) == cfg.rerank.page_size


def test_invalid_country_and_event_type_422(client):
    assert _feed(client, "u_0", "ZZ").status_code == 422
    r = client.post("/events", json={"events": [{"event_id": "x", "user_id": "u_0", "product_id": "p_1", "type": "teleport", "ts": NOW}]})
    assert r.status_code == 422


def test_events_ingest_is_idempotent(client):
    ev = {"event_id": "dup1", "user_id": "u_0", "product_id": "p_1", "type": "view", "ts": NOW}
    assert client.post("/events", json={"events": [ev]}).json() == {"accepted": 1, "duplicates": 0, "ignored_future": 0}
    assert client.post("/events", json={"events": [ev]}).json() == {"accepted": 0, "duplicates": 1, "ignored_future": 0}


def test_feed_logs_one_impression_per_card(client, service):
    before = len(service.logger)
    body = _feed(client, "u_1", "US").json()
    log = service.logger.to_frame().tail(len(body["cards"]))
    assert len(service.logger) == before + len(body["cards"])
    assert (log.model_version == service.model_version).all()
    assert sorted(log.position) == list(range(len(body["cards"])))
    assert (log.request_id == body["request_id"]).all()
    assert all(len(f) > 0 for f in log.features)


def test_country_with_no_eligible_products_gets_fallback_or_empty(world, trained, cfg):
    tables = {k: v.copy() for k, v in world.items()}
    tables["products"]["ships_to"] = [[c for c in s if c != "DE"] for s in tables["products"].ships_to]
    svc = make_service(tables, trained, cfg)
    resp = svc.recommend(FeedRequest(user_id="u_0", country="DE", session_id="s", now=NOW))
    assert resp.fallback and resp.cards == []


def test_fallback_when_personalized_sources_empty(world, trained, cfg):
    """No index, no follows, no fresh, no trending -> fallback popularity page, still eligible-only."""
    tables = {k: v.copy() for k, v in world.items()}
    tables["products"]["published_at"] = NOW - 5 * DAY
    svc = make_service(tables, trained, cfg)
    svc.index.ids = svc.index.ids[:0]; svc.index.vecs = svc.index.vecs[:0]
    svc.follows = {}
    from shopfeed.features.streaming import StreamingStore
    svc.streaming = StreamingStore()
    resp = svc.recommend(FeedRequest(user_id="ghost", country="US", session_id="s", now=NOW))
    assert resp.fallback and len(resp.cards) > 0
    assert all(c.source == "fallback" for c in resp.cards)


# ---------------------------------------------------------------- edge cases
def test_brand_new_user_is_cold_and_still_served(service, cfg):
    ctx = service.build_context("brand_new", "CA", NOW)
    assert ctx.is_cold and np.allclose(ctx.user_vec, service._prior_vec)
    resp = service.recommend(FeedRequest(user_id="brand_new", country="CA", session_id="s", now=NOW))
    assert len(resp.cards) == cfg.rerank.page_size


def test_shop_pay_only_user_is_personalized(service, world):
    u = _user(world, "shop_pay_only")
    ctx = service.build_context(u.user_id, u.country, NOW)
    assert not ctx.is_cold
    assert not np.allclose(ctx.user_vec, service._prior_vec)
    assert len(ctx.purchased_categories) > 0


def test_brand_new_product_enters_via_fresh_lane(service, world, cfg):
    from shopfeed.retrieval.sources import UserContext, eligibility_mask, source_fresh
    new = service.products.iloc[[0]].copy()
    new["product_id"] = "p_new"; new["published_at"] = NOW - 60; new["in_stock"] = True; new["ships_to"] = [["US", "CA", "GB", "AU", "DE"]]
    new["merchant_status"] = "active"
    prods = pd.concat([service.products, new], ignore_index=True)
    ctx = UserContext(user_id="u_0", country="US", now=NOW)
    out = source_fresh(ctx, prods, eligibility_mask(prods, ctx, cfg.eligibility), cfg.caps.fresh)
    assert out.product_id.iloc[0] == "p_new"


def test_recent_future_events_ignored_by_service(service):
    ev = pd.DataFrame([{"event_id": "fut", "user_id": "u_0", "product_id": "p_1", "merchant_id": "m_0", "type": "view", "ts": NOW + 10_000, "impression_id": None, "value": np.nan}])
    assert service.ingest_events(ev, now=NOW)["ignored_future"] == 1


# ---------------------------------------------------------------- business cases
def _inject(service, user_id, product_id, etype, ts):
    prod = service._prod_by_id.loc[product_id]
    ev = pd.DataFrame([{"event_id": f"inj_{user_id}_{product_id}_{etype}_{ts}", "user_id": user_id, "product_id": product_id,
                        "merchant_id": prod.merchant_id, "type": etype, "ts": ts, "impression_id": None, "value": np.nan}])
    service.ingest_events(ev, now=max(NOW, ts))


def _candidates(service, user_id, country):
    from shopfeed.retrieval.sources import generate_candidates
    ctx = service.build_context(user_id, country, NOW)
    return ctx, generate_candidates(ctx, service.products, service.index, service.streaming, service.popularity, service.cfg.caps, service.cfg.eligibility)


def test_shoe_buyer_gets_socks_and_insoles_as_complements_not_more_shoes(service):
    shoes = service.products[(service.products.category == "running_shoes") & service.products.in_stock & (service.products.merchant_status == "active")].product_id.iloc[0]
    _inject(service, "buyer", shoes, "purchase", NOW - 10 * DAY)
    ctx, cands = _candidates(service, "buyer", "US")
    comp = cands[cands.sources.apply(lambda s: "complements" in s)]
    cats = set(service._prod_by_id.loc[comp.product_id].category)
    assert len(comp) > 0 and cats <= {"socks", "insoles"}
    page = service.recommend(FeedRequest(user_id="buyer", country="US", session_id="s", now=NOW))
    assert shoes not in {c.product_id for c in page.cards}  # not recommended again


def test_consumable_due_for_repurchase_is_a_candidate_and_recent_one_is_not(service):
    coffee = service.products[(service.products.category == "coffee") & service.products.in_stock & (service.products.merchant_status == "active")
                              & service.products.ships_to.apply(lambda s: "US" in s)]
    due, fresh = coffee.product_id.iloc[0], coffee.product_id.iloc[1]
    _inject(service, "drinker", due, "purchase", NOW - 28 * DAY)
    _inject(service, "drinker", fresh, "purchase", NOW - 3 * DAY)
    _, cands = _candidates(service, "drinker", "US")
    rep = cands[cands.sources.apply(lambda s: "replenish" in s)]
    assert due in set(rep.product_id)
    assert fresh not in set(cands.product_id)


def test_followed_merchants_drop_published_minutes_ago_is_candidate(service):
    m = service.products[service.products.merchant_status == "active"].merchant_id.iloc[0]
    new = service.products[service.products.merchant_id == m].iloc[[0]].copy()
    new["product_id"] = "p_drop"; new["published_at"] = NOW - 300; new["in_stock"] = True; new["ships_to"] = [["US", "CA", "GB", "AU", "DE"]]
    assert service.add_products(new) == 1 and "p_drop" in service.index.ids
    assert service.add_products(new) == 0  # idempotent
    service.follows["fan"] = {m}
    _, cands = _candidates(service, "fan", "US")
    row = cands[cands.product_id == "p_drop"]
    assert len(row) == 1 and {"followed", "fresh"} <= set(row.sources.iloc[0])
    page = service.recommend(FeedRequest(user_id="fan", country="US", session_id="s", now=NOW))
    assert len(page.cards) == service.cfg.rerank.page_size  # index and catalog stay consistent after publish


def test_hidden_product_never_reappears(service, cfg):
    first = service.recommend(FeedRequest(user_id="u_2", country="US", session_id="s", now=NOW))
    target = first.cards[0].product_id
    _inject(service, "u_2", target, "hide", NOW + 1)
    service.shown["u_2"] = {}  # clear the 7-day repeat block so only the hide can exclude it
    again = service.recommend(FeedRequest(user_id="u_2", country="US", session_id="s", now=NOW + 2))
    assert target not in {c.product_id for c in again.cards}


def test_repeat_block_prevents_showing_same_item_within_window(service):
    a = service.recommend(FeedRequest(user_id="u_3", country="US", session_id="s", now=NOW))
    b = service.recommend(FeedRequest(user_id="u_3", country="US", session_id="s", now=NOW + 60))
    assert not ({c.product_id for c in a.cards} & {c.product_id for c in b.cards})


def test_head_merchant_share_under_cap_on_every_page(service, world, cfg):
    cap = int(cfg.rerank.merchant_share_cap * cfg.rerank.page_size)
    for u in world["users"].head(15).itertuples():
        page = service.recommend(FeedRequest(user_id=u.user_id, country=u.country, session_id="s", now=NOW))
        if page.cards:
            assert pd.Series([c.merchant_id for c in page.cards]).value_counts().max() <= cap


def test_exploration_slots_logged_with_propensity_and_ips_weightable(service, cfg):
    for i in range(5):
        service.recommend(FeedRequest(user_id=f"u_{i}", country="US", session_id="s", now=NOW))
    log = service.logger.to_frame()
    expl = log[log.propensity < 1.0]
    assert len(expl) == 5 * cfg.rerank.exploration_slots
    reward = np.random.default_rng(0).random(len(expl))
    est = ips_estimate(reward, expl.propensity.values, np.ones(len(expl)))
    assert np.isfinite(est)


def test_clickbait_ranks_lower_once_hide_and_return_weights_apply(trained, world, cfg):
    """Ablation: with hide/return weights zeroed the clickbait cluster moves UP the ranking."""
    ts = trained["ts"]
    P = trained["heavy"].predict_proba(ts.X)
    prod = ts.meta[["product_id"]].merge(world["products"][["product_id", "price", "is_clickbait"]], on="product_id", how="left")
    cb = prod.is_clickbait.values.astype(bool)
    assert cb.sum() > 20
    v_default = value_score(P, prod.price.values, cfg.value)
    v_blind = value_score(P, prod.price.values, ValueWeights(hide=0.0, return_=0.0))
    rank_default = pd.Series(v_default).rank(ascending=False).values
    rank_blind = pd.Series(v_blind).rank(ascending=False).values
    assert rank_default[cb].mean() > rank_blind[cb].mean()           # clickbait pushed down by the penalties
    assert P[cb, HEADS.index("hide")].mean() > P[~cb, HEADS.index("hide")].mean()  # model learned the hide signal


def test_ab_model_beats_random_policy_on_simulated_gmv(world, trained, cfg):
    users = world["users"][world["users"].cohort == "normal"].head(90)
    rng = np.random.default_rng(0)
    rows = []
    for policy in ("model", "random"):
        svc = make_service(world, trained, cfg, seed=1, policy=policy)
        for u in users.itertuples(index=False):
            resp = svc.recommend(FeedRequest(user_id=u.user_id, country=u.country, session_id="ab", now=NOW))
            page = pd.DataFrame([c.model_dump() for c in resp.cards])
            sim = simulate_responses(page, users.set_index("user_id").loc[u.user_id], world["products"], world["merchants"], rng)
            rows.append({"user_id": f"{policy}_{u.user_id}", "bucket": "treatment" if policy == "model" else "control", "metric": sim.order_value.sum()})
    r = compare(pd.DataFrame(rows), n_boot=300)
    assert r["lift"] > 0
    assert r["treatment_mean"] > r["control_mean"] * 1.2
