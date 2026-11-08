import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from shopfeed.data.synth import generate
from shopfeed.schema import Product


def test_deterministic_for_seed():
    a = generate(seed=3, n_users=60, n_merchants=6, n_products=120, sessions_per_user=3)
    b = generate(seed=3, n_users=60, n_merchants=6, n_products=120, sessions_per_user=3)
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])


def test_every_product_has_a_merchant(world):
    assert world["products"].merchant_id.isin(world["merchants"].merchant_id).all()


def test_event_rates_in_expected_ranges(world):
    n_imp = len(world["impressions"])
    ev = world["events"]
    click_rate = (ev.type == "view").sum() / n_imp
    purchase_rate = (ev.type == "purchase").sum() / n_imp
    assert 0.05 < click_rate < 0.3
    assert 0.005 < purchase_rate < 0.08
    assert ev.event_id.is_unique


def test_clickbait_cluster_has_higher_hide_rate(world):
    imp = world["impressions"].merge(world["products"][["product_id", "is_clickbait"]], on="product_id")
    hides = world["events"][world["events"].type == "hide"].impression_id
    imp["hidden"] = imp.impression_id.isin(hides)
    rate = imp.groupby("is_clickbait").hidden.mean()
    assert rate[True] > rate[False]


def test_cold_users_have_no_events(world):
    cold = world["users"][world["users"].cohort == "cold"].user_id
    assert not world["events"].user_id.isin(cold).any()


def test_shop_pay_only_users_have_purchases_without_impressions(world):
    sp = world["users"][world["users"].cohort == "shop_pay_only"].user_id
    ev = world["events"][world["events"].user_id.isin(sp)]
    assert (ev.type == "purchase").all()
    assert ev.impression_id.isna().all()


def test_cold_products_exist(world):
    p = world["products"]
    assert ((1_750_000_000.0 - p.published_at) < 3600).any()


@pytest.mark.parametrize("price", [0, -5.0])
def test_product_price_must_be_positive(price):
    with pytest.raises(ValidationError):
        Product(product_id="p", merchant_id="m", category="socks", price=price, ships_to=["US"], published_at=0.0)


def test_product_unknown_category_rejected():
    with pytest.raises(ValidationError):
        Product(product_id="p", merchant_id="m", category="yachts", price=9.0, ships_to=["US"], published_at=0.0)
