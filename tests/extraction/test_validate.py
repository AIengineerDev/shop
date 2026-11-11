from decimal import Decimal

from repricer.extraction.validate import amount_in_source, validate
from repricer.models import ExtractedOffer, OfferStatus, ReasonCode


def offer(**kw: object) -> ExtractedOffer:
    base: dict[str, object] = {
        "scrape_id": 1,
        "listed_price": Decimal("42.99"),
        "currency": "USD",
        "confidence": 80,
    }
    base.update(kw)
    return ExtractedOffer(**base)  # type: ignore[arg-type]


TEXT = "Widget on sale for $42.99 with $5 coupon"


def test_ac_e2_fabricated_price_rejected() -> None:
    v = validate(
        offer(listed_price=Decimal("41.99")), TEXT, resolved_sku="S", current_price_cents=4500
    )
    assert v.status is OfferStatus.REJECTED and v.reject_reason is ReasonCode.PRICE_NOT_IN_SOURCE


def test_fabricated_coupon_rejected() -> None:
    v = validate(
        offer(coupon_amount=Decimal("7")), TEXT, resolved_sku="S", current_price_cents=4500
    )
    assert v.reject_reason is ReasonCode.COUPON_NOT_IN_SOURCE
    v = validate(
        offer(coupon_percent=Decimal("15")), TEXT, resolved_sku="S", current_price_cents=4500
    )
    assert v.reject_reason is ReasonCode.COUPON_NOT_IN_SOURCE


def test_good_offer_accepted() -> None:
    v = validate(
        offer(coupon_amount=Decimal("5")), TEXT, resolved_sku="S", current_price_cents=4500
    )
    assert v.status is OfferStatus.ACCEPTED and v.codes == []


def test_amount_forms() -> None:
    assert amount_in_source(Decimal("1042.99"), "now $1,042.99")
    assert amount_in_source(Decimal("42.99"), "42,99 EUR")
    assert amount_in_source(Decimal("5"), "$5 coupon")
    assert not amount_in_source(Decimal("2.99"), "$42.99")  # substring is not a match
    assert not amount_in_source(Decimal("42.9"), "$42.99")


def test_zero_and_implausible() -> None:
    v = validate(
        offer(listed_price=Decimal("0")), "price $0", resolved_sku="S", current_price_cents=4500
    )
    assert v.reject_reason is ReasonCode.ZERO_PRICE
    v = validate(
        offer(listed_price=Decimal("99999")), "$99999", resolved_sku="S", current_price_cents=4500
    )
    assert v.reject_reason is ReasonCode.IMPLAUSIBLE_PRICE


def test_sku_unresolved() -> None:
    v = validate(offer(), TEXT, resolved_sku=None, current_price_cents=None)
    assert v.reject_reason is ReasonCode.SKU_UNRESOLVED


def test_needs_review_cases() -> None:
    v = validate(
        offer(coupon_amount=Decimal("5"), coupon_percent=Decimal("10")),
        "$42.99: $5 off OR 10% off",
        resolved_sku="S",
        current_price_cents=4500,
    )
    assert v.status is OfferStatus.NEEDS_REVIEW and v.reject_reason is ReasonCode.AMBIGUOUS_COUPON
    v = validate(offer(is_bundle=True), "2 for $42.99", resolved_sku="S", current_price_cents=4500)
    assert v.status is OfferStatus.NEEDS_REVIEW and v.reject_reason is ReasonCode.BUNDLE_PRICE


def test_non_usd_accepted_but_flagged() -> None:
    v = validate(offer(currency="EUR"), "€42.99", resolved_sku="S", current_price_cents=4500)
    assert v.status is OfferStatus.ACCEPTED and v.codes == [ReasonCode.NON_USD]
