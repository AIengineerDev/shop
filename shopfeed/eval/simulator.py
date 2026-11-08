"""Replays the synthetic world model against a served page to simulate user responses.

Used for: offline guardrail estimates of a NEW ranker (which has no logged traffic yet),
A/B simulations, and the business-case tests. It is the same formula the data came from,
so it is an oracle, not a claim about real users.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from shopfeed.data.synth import CONTENT_DIM, latent_engagement

TASTE_COLS = [f"t{d}" for d in range(CONTENT_DIM)]
CONTENT_COLS = [f"c{d}" for d in range(CONTENT_DIM)]


def simulate_responses(page: pd.DataFrame, user_row: pd.Series, products: pd.DataFrame, merchants: pd.DataFrame,
                       rng: np.random.Generator) -> pd.DataFrame:
    """page: columns product_id, position. Returns page + boolean response columns + order_value."""
    if len(page) == 0:
        return page.assign(click=False, dwell=False, save=False, atc=False, purchase=False, hide=False, **{"return": False}, order_value=0.0)
    prod = page[["product_id"]].merge(products, on="product_id", how="left")
    merch = prod[["merchant_id"]].merge(merchants, on="merchant_id", how="left")
    taste = user_row[TASTE_COLS].values.astype(float)
    content = prod[CONTENT_COLS].values.astype(float)
    aff = np.dot(content, taste) / np.sqrt(CONTENT_DIM)
    logp = np.log(products.price.values.astype(float))
    price_z = (np.log(prod.price.values.astype(float)) - logp.mean()) / logp.std()
    eng = latent_engagement(aff, prod.is_clickbait.values.astype(float), price_z,
                            merch.review_score.values.astype(float) - 4.3, merch.return_rate.values.astype(float),
                            page.position.values, rng)
    out = page.copy()
    for k, v in eng.items():
        out[k] = v
    out["order_value"] = np.where(eng["purchase"], prod.price.values.astype(float), 0.0)
    return out


def responses_to_events(resp: pd.DataFrame, user_id: str, now: float, start_id: int = 0) -> pd.DataFrame:
    rows = []
    i = start_id
    type_map = {"click": "view", "dwell": "dwell", "save": "save", "atc": "add_to_cart", "purchase": "purchase", "hide": "hide", "return": "return"}
    for r in resp.to_dict("records"):  # "return" is a keyword, so itertuples would rename it
        for col, et in type_map.items():
            if r[col]:
                rows.append({"event_id": f"sim_{i}", "user_id": user_id, "product_id": r["product_id"], "merchant_id": r["merchant_id"],
                             "type": et, "ts": now + 1.0, "impression_id": r["impression_id"], "value": r["order_value"] if et == "purchase" else np.nan})
                i += 1
    return pd.DataFrame(rows, columns=["event_id", "user_id", "product_id", "merchant_id", "type", "ts", "impression_id", "value"])
