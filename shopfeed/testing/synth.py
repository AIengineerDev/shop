"""Synthetic feed log with the same parquet schema as data/ (integration, e2e and load tests).

Signal: P(click) = examination(position) * sigmoid(a * cos(user, item) + b). Clicked items
convert to wishlist / add-to-cart / purchase with fixed rates. Positions are randomised
(like the real log), so the examination curve is recoverable. Purchases arrive 1 min - 48 h
after the impression with a stamped impression_id; a share of purchases have no impression
(non-search). Rates are tuned so the §9 readiness checks pass with the "ci" profile.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["mugs", "socks", "coffee", "candles", "skincare", "headphones", "jeans", "tshirts"]
T0 = 1_746_000_000.0
DAY = 86400.0


def generate(out_dir: Path, n_users: int = 300, n_products: int = 400, n_merchants: int = 20,
             n_requests: int = 1500, page: int = 20, days: int = 30, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    users = pd.DataFrame({"user_id": [f"u_{i}" for i in range(n_users)],
                          "country": "US", "cohort": "normal"})
    ut = rng.normal(size=(n_users, 8))
    for k in range(8):
        users[f"t{k}"] = ut[:, k]

    merchants = pd.DataFrame({
        "merchant_id": [f"m_{i}" for i in range(n_merchants)], "status": "active",
        "fulfilment_days": rng.uniform(0.5, 4, n_merchants).round(2),
        "return_rate": rng.uniform(0.02, 0.3, n_merchants).round(4),
        "review_score": rng.uniform(3.5, 5, n_merchants).round(2),
        "popularity": rng.uniform(0, 1, n_merchants)})

    pc = rng.normal(size=(n_products, 8))
    products = pd.DataFrame({
        "product_id": [f"p_{i}" for i in range(n_products)],
        "merchant_id": [f"m_{i}" for i in rng.integers(0, n_merchants, n_products)],
        "category": rng.choice(CATEGORIES, n_products),
        "price": rng.uniform(8, 300, n_products).round(2), "currency": "USD", "in_stock": True,
        "published_at": T0 + rng.uniform(-20, days, n_products) * DAY})
    for k in range(8):
        products[f"c{k}"] = pc[:, k]

    un = ut / np.linalg.norm(ut, axis=1, keepdims=True)
    pn = pc / np.linalg.norm(pc, axis=1, keepdims=True)

    imp_rows, ev_rows = [], []
    req_ts = np.sort(T0 + rng.uniform(0, days * DAY, n_requests))
    ev_id = 0

    def event(user, prod, typ, ts, imp_id, value=np.nan):
        nonlocal ev_id
        ev_rows.append((f"ev_{ev_id}", user, prod, products.at[prod_idx[prod], "merchant_id"],
                        typ, ts, imp_id, value))
        ev_id += 1

    prod_idx = {p: i for i, p in enumerate(products["product_id"])}
    for r, ts in enumerate(req_ts):
        u = int(rng.integers(n_users))
        items = rng.choice(n_products, page, replace=False)
        for pos, it in enumerate(items):  # positions are random w.r.t. relevance
            imp_id = f"imp_{r}_{pos}"
            user, prod = f"u_{u}", f"p_{it}"
            t = ts + rng.uniform(0, 5)
            imp_rows.append((imp_id, f"req_{r}", user, prod, "product", t, pos, 1 / page,
                             "logging_v0", "logging_random"))
            exam = 1.0 / (1.0 + 0.12 * pos)
            p_click = exam / (1 + np.exp(-(3.0 * float(un[u] @ pn[it]) - 1.6)))
            if rng.random() >= p_click:
                continue
            event(user, prod, "view", t + rng.uniform(1, 60), imp_id)
            if rng.random() < 0.05:
                event(user, prod, "hide", t + rng.uniform(1, 30), imp_id)
                continue
            if rng.random() < 0.2:
                event(user, prod, "save", t + rng.uniform(10, 300), imp_id)
            if rng.random() < 0.12:
                event(user, prod, "add_to_cart", t + rng.uniform(20, 600), imp_id)
            if rng.random() < 0.13:
                price = float(products.at[it, "price"])
                pt = t + rng.uniform(60, 2 * DAY)
                event(user, prod, "purchase", pt, imp_id, price)
                if rng.random() < 0.1:
                    event(user, prod, "return", pt + rng.uniform(3, 12) * DAY, imp_id)
        if rng.random() < 0.1:  # non-search purchase, no impression
            it = int(rng.integers(n_products))
            event(f"u_{u}", f"p_{it}", "purchase", ts + rng.uniform(0, DAY), None,
                  float(products.at[it, "price"]))
        if rng.random() < 0.05:
            m = f"m_{rng.integers(n_merchants)}"
            ev_rows.append((f"ev_{ev_id}", f"u_{u}", None, m, "follow", ts, None, np.nan))
            ev_id += 1

    impressions = pd.DataFrame(imp_rows, columns=[
        "impression_id", "request_id", "user_id", "product_id", "card_type", "ts", "position",
        "propensity", "model_version", "source"])
    events = pd.DataFrame(ev_rows, columns=[
        "event_id", "user_id", "product_id", "merchant_id", "type", "ts", "impression_id",
        "value"])
    for name, df in [("users", users), ("products", products), ("merchants", merchants),
                     ("impressions", impressions), ("events", events)]:
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    meta = {"now": float(T0 + (days + 5) * DAY), "seed": seed}
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return {"impressions": len(impressions), "events": len(events), "requests": n_requests}


if __name__ == "__main__":
    import sys

    print(generate(Path(sys.argv[1])))
