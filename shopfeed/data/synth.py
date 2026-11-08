"""Seeded synthetic Shop-app world.

Latent user taste vectors x product content vectors drive engagement, so models have a real
signal to learn. The generator bakes in the phenomena the system claims to handle:
Zipf merchant popularity, a clickbait cluster (high click / high hide / high return),
consumables, position bias, shipping restrictions, out-of-stock, cold users, Shop-Pay-only
users (purchases without app impressions) and cold products.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from shopfeed.config import (CATEGORIES, CONSUMABLE_CYCLE_DAYS, COUNTRIES, DAY, HOUR)

CONTENT_DIM = 8
LOGGING_MODEL_VERSION = "logging_v0"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def position_decay(pos: np.ndarray) -> np.ndarray:
    """Attention falls with slot. Same curve the position-bias tests assume."""
    return 1.0 / (1.0 + 0.08 * np.asarray(pos, dtype=float))


def latent_engagement(aff, cb, price_z, mq, m_ret, pos, rng) -> Dict[str, np.ndarray]:
    """The world model. aff = taste.content/sqrt(d); cb = clickbait flag; price_z = log-price z-score;
    mq = merchant review score - 4.3; m_ret = merchant return rate; pos = slot."""
    n = len(aff)
    decay = position_decay(pos)
    p_click = _sigmoid(-1.8 + 1.2 * aff + 1.5 * cb) * decay
    click = rng.random(n) < p_click
    dwell = click & (rng.random(n) < _sigmoid(-0.3 + 1.0 * aff - 1.2 * cb))
    save = click & (rng.random(n) < _sigmoid(-2.0 + 1.0 * aff - 0.5 * cb))
    atc = click & (rng.random(n) < _sigmoid(-2.0 + 0.9 * aff - 0.3 * price_z - 0.5 * cb))
    purchase = atc & (rng.random(n) < _sigmoid(-0.3 + 0.6 * aff - 0.5 * price_z + 0.8 * mq))
    hide = rng.random(n) < _sigmoid(-4.5 - 1.0 * aff + 3.0 * cb) * decay
    ret = purchase & (rng.random(n) < np.clip(m_ret * 1.0 + 0.3 * cb, 0, 0.9))
    return {"click": click, "dwell": dwell, "save": save, "atc": atc, "purchase": purchase, "hide": hide, "return": ret}


def category_centroids(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0, 1, size=(len(CATEGORIES), CONTENT_DIM))


def generate(
    seed: int = 42,
    n_users: int = 2000,
    n_merchants: int = 60,
    n_products: int = 4000,
    sessions_per_user: float = 6.0,
    page_size: int = 20,
    history_days: float = 45.0,
    now: float = 1_750_000_000.0,
) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    centroids = category_centroids(rng)

    # ---------------- merchants ----------------
    ranks = np.arange(1, n_merchants + 1)
    pop = 1.0 / ranks ** 1.1
    pop = pop / pop.sum()
    merchants = pd.DataFrame({
        "merchant_id": [f"m_{i}" for i in range(n_merchants)],
        "status": np.where(rng.random(n_merchants) < 0.03, "suspended", "active"),
        "fulfilment_days": np.clip(rng.normal(3, 1.5, n_merchants), 0.5, 14).round(2),
        "return_rate": np.clip(rng.beta(2, 20, n_merchants), 0, 1).round(4),
        "review_score": np.clip(rng.normal(4.3, 0.4, n_merchants), 1, 5).round(2),
        "popularity": pop,
    })

    # ---------------- products ----------------
    merchant_idx = rng.choice(n_merchants, size=n_products, p=pop)
    cat_idx = rng.integers(0, len(CATEGORIES), size=n_products)
    cats = np.array(CATEGORIES)[cat_idx]
    content = centroids[cat_idx] + rng.normal(0, 0.5, size=(n_products, CONTENT_DIM))
    base_price = {c: 20 + 10 * i for i, c in enumerate(CATEGORIES)}
    price = np.array([base_price[c] for c in cats]) * np.exp(rng.normal(0, 0.4, n_products))
    ships_all = rng.random(n_products) < 0.7
    ships_to: List[List[str]] = []
    for i in range(n_products):
        if ships_all[i]:
            ships_to.append(list(COUNTRIES))
        else:
            k = rng.integers(1, len(COUNTRIES))
            ships_to.append(sorted(rng.choice(COUNTRIES, size=k, replace=False).tolist()))
    published = now - rng.uniform(0, 60 * DAY, n_products)
    cold = rng.random(n_products) < 0.02
    published[cold] = now - rng.uniform(60, HOUR, cold.sum())
    is_consumable = np.array([c in CONSUMABLE_CYCLE_DAYS for c in cats])
    cycle = np.array([CONSUMABLE_CYCLE_DAYS.get(c, np.nan) * DAY for c in cats])
    is_clickbait = rng.random(n_products) < 0.03
    products = pd.DataFrame({
        "product_id": [f"p_{i}" for i in range(n_products)],
        "merchant_id": merchants.merchant_id.values[merchant_idx],
        "category": cats,
        "price": price.round(2),
        "currency": "USD",
        "in_stock": rng.random(n_products) < 0.9,
        "ships_to": ships_to,
        "published_at": published,
        "is_consumable": is_consumable,
        "repurchase_cycle_s": cycle,
        "is_evergreen": rng.random(n_products) < 0.1,
        "is_clickbait": is_clickbait,  # latent; never a serving feature
    })
    for d in range(CONTENT_DIM):
        products[f"c{d}"] = content[:, d].round(4)

    # ---------------- users ----------------
    cohort = rng.choice(["normal", "cold", "shop_pay_only"], size=n_users, p=[0.8, 0.1, 0.1])
    taste = rng.normal(0, 1, size=(n_users, CONTENT_DIM))
    blocked = []
    for i in range(n_users):
        k = rng.choice([0, 0, 0, 1, 2])
        blocked.append(sorted(rng.choice(merchants.merchant_id.values, size=k, replace=False).tolist()))
    users = pd.DataFrame({
        "user_id": [f"u_{i}" for i in range(n_users)],
        "country": rng.choice(COUNTRIES, size=n_users),
        "blocked_merchants": blocked,
        "shop_pay_history": (cohort == "shop_pay_only") | ((cohort == "normal") & (rng.random(n_users) < 0.4)),
        "cohort": cohort,  # latent, for tests
    })
    for d in range(CONTENT_DIM):
        users[f"t{d}"] = taste[:, d].round(4)

    # ---------------- follows ----------------
    f_users, f_merch = [], []
    for i in np.where(cohort == "normal")[0]:
        k = rng.integers(0, 6)
        for m in rng.choice(n_merchants, size=k, replace=False, p=pop):
            f_users.append(users.user_id[i])
            f_merch.append(merchants.merchant_id[m])
    follows = pd.DataFrame({"user_id": f_users, "merchant_id": f_merch}).drop_duplicates()

    # ---------------- impressions from a popularity-tilted random logging policy ----------------
    normal_idx = np.where(cohort == "normal")[0]
    n_sessions = rng.poisson(sessions_per_user, size=len(normal_idx))
    sess_user = np.repeat(normal_idx, n_sessions)
    n_sess_total = len(sess_user)
    sess_ts = now - rng.uniform(0, history_days * DAY, n_sess_total)
    imp_user = np.repeat(sess_user, page_size)
    imp_ts = np.repeat(sess_ts, page_size) + np.tile(np.arange(page_size) * 2.0, n_sess_total)
    imp_pos = np.tile(np.arange(page_size), n_sess_total)
    imp_req = np.repeat(np.arange(n_sess_total), page_size)
    # Only products that exist at impression time, in stock, from active merchants, shipping to the user.
    merch_pop = merchants.set_index("merchant_id").popularity
    item_w = np.sqrt(products.merchant_id.map(merch_pop).values)
    item_w = item_w / item_w.sum()
    n_imp = len(imp_user)
    imp_item = rng.choice(n_products, size=n_imp, p=item_w)
    # Enforce "existed at impression time" by resampling the few that violate it. Products published in
    # the last hour are the cold-start cohort: by definition they have no interactions yet.
    def _bad(items):
        return (products.published_at.values[items] > imp_ts) | cold[items]
    bad = _bad(imp_item)
    while bad.any():
        imp_item[bad] = rng.choice(n_products, size=bad.sum(), p=item_w)
        bad = _bad(imp_item)

    # ---------------- latent engagement ----------------
    aff = np.einsum("ij,ij->i", taste[imp_user], content[imp_item])
    aff = aff / np.sqrt(CONTENT_DIM)
    cb = is_clickbait[imp_item].astype(float)
    price_z = (np.log(price[imp_item]) - np.log(price).mean()) / np.log(price).std()
    mq = merchants.set_index("merchant_id").review_score.reindex(products.merchant_id.values[imp_item]).values - 4.3
    m_ret = merchants.set_index("merchant_id").return_rate.reindex(products.merchant_id.values[imp_item]).values
    decay = position_decay(imp_pos)

    eng = latent_engagement(aff, cb, price_z, mq, m_ret, imp_pos, rng)
    click, dwell, save, atc, purchase, hide, ret = (eng[k] for k in ("click", "dwell", "save", "atc", "purchase", "hide", "return"))

    impressions = pd.DataFrame({
        "impression_id": [f"imp_{i}" for i in range(n_imp)],
        "request_id": [f"req_{r}" for r in imp_req],
        "user_id": users.user_id.values[imp_user],
        "product_id": products.product_id.values[imp_item],
        "card_type": "product",
        "ts": imp_ts,
        "position": imp_pos,
        "propensity": item_w[imp_item] * page_size,  # P(item in page) under the logging policy
        "model_version": LOGGING_MODEL_VERSION,
        "source": "logging_random",
    })

    # ---------------- events ----------------
    ev_frames = []
    def _emit(mask: np.ndarray, etype: str, delay_lo: float, delay_hi: float, value=None):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        d = rng.uniform(delay_lo, delay_hi, len(idx))
        ev_frames.append(pd.DataFrame({
            "user_id": users.user_id.values[imp_user[idx]],
            "product_id": products.product_id.values[imp_item[idx]],
            "merchant_id": products.merchant_id.values[imp_item[idx]],
            "type": etype,
            "ts": imp_ts[idx] + d,
            "impression_id": impressions.impression_id.values[idx],
            "value": value[idx] if value is not None else np.nan,
        }))

    _emit(click, "view", 1, 60)
    _emit(dwell, "dwell", 5, 120, value=rng.uniform(8, 90, n_imp))
    _emit(save, "save", 10, 300)
    _emit(atc, "add_to_cart", 20, 600)
    _emit(purchase, "purchase", 60, 2 * DAY, value=price[imp_item])
    _emit(hide, "hide", 1, 30)
    _emit(ret, "return", 3 * DAY, 12 * DAY)  # some land outside the 7d label window on purpose

    # Shop-Pay-only users: purchases with no impression (checked out on merchant storefronts).
    sp_idx = np.where(cohort == "shop_pay_only")[0]
    sp_rows = []
    for u in sp_idx:
        k = rng.integers(3, 9)
        # Purchases follow taste so personalization from Shop Pay history is learnable.
        scores = np.dot(content, taste[u]) / np.sqrt(CONTENT_DIM)
        p = np.exp(scores) * (~cold); p /= p.sum()
        items = rng.choice(n_products, size=k, replace=False, p=p)
        for it in items:
            lo = max(published[it], now - 60 * DAY)
            sp_rows.append((users.user_id[u], products.product_id[it], products.merchant_id[it],
                            "purchase", rng.uniform(lo, now), None, price[it]))
    if sp_rows:
        ev_frames.append(pd.DataFrame(sp_rows, columns=["user_id", "product_id", "merchant_id", "type", "ts", "impression_id", "value"]))

    # Follow events mirror the follows table.
    if len(follows):
        ev_frames.append(pd.DataFrame({
            "user_id": follows.user_id.values, "product_id": None, "merchant_id": follows.merchant_id.values,
            "type": "follow", "ts": now - rng.uniform(0, history_days * DAY, len(follows)),
            "impression_id": None, "value": np.nan,
        }))

    events = pd.concat(ev_frames, ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
    events.insert(0, "event_id", [f"ev_{i}" for i in range(len(events))])
    events["impression_id"] = events["impression_id"].astype(object)
    events["product_id"] = events["product_id"].astype(object)

    return {
        "users": users, "merchants": merchants, "products": products, "follows": follows,
        "impressions": impressions.sort_values("ts", kind="stable").reset_index(drop=True),
        "events": events,
    }
