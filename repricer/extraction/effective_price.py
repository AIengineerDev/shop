"""Spec §4.4: effective competitor price, computed in code from extracted entities."""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal

from repricer.models import ExtractedOffer, ReasonCode
from repricer.money import Cents, to_cents_rounded

CENT = Decimal("0.01")


def effective_price(offer: ExtractedOffer) -> tuple[Cents | None, list[ReasonCode]]:
    if offer.listed_price is None:
        return None, []
    codes: list[ReasonCode] = []
    price = offer.listed_price
    if offer.coupon_amount is not None:
        price -= offer.coupon_amount
    if offer.coupon_percent is not None:
        pct_off = (offer.listed_price * offer.coupon_percent / 100).quantize(
            CENT, rounding=ROUND_HALF_EVEN
        )
        price -= pct_off
    if offer.shipping is not None:
        price += offer.shipping
    else:
        codes.append(ReasonCode.SHIPPING_UNKNOWN)
    if price < 0:
        price = Decimal(0)
    return to_cents_rounded(price), codes
