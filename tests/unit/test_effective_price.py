from decimal import Decimal

from repricer.extraction.effective_price import effective_price
from repricer.models import ExtractedOffer, ReasonCode


def offer(**kw: object) -> ExtractedOffer:
    base: dict[str, object] = {"scrape_id": 1, "listed_price": Decimal("42.99"), "confidence": 80}
    base.update(kw)
    return ExtractedOffer(**base)  # type: ignore[arg-type]


def test_coupon_amount() -> None:
    eff, codes = effective_price(offer(coupon_amount=Decimal("5")))
    assert eff == 3799
    assert codes == [ReasonCode.SHIPPING_UNKNOWN]


def test_coupon_percent_bankers_rounding() -> None:
    # 10% of 42.99 = 4.299 -> 4.30 ; 42.99 - 4.30 = 38.69
    eff, _ = effective_price(offer(coupon_percent=Decimal("10"), shipping=Decimal("0")))
    assert eff == 3869


def test_shipping_added_and_known() -> None:
    eff, codes = effective_price(offer(shipping=Decimal("4.99")))
    assert eff == 4798
    assert codes == []


def test_free_shipping_is_known_zero() -> None:
    eff, codes = effective_price(offer(shipping=Decimal("0")))
    assert eff == 4299 and codes == []


def test_never_negative() -> None:
    eff, _ = effective_price(offer(coupon_amount=Decimal("100")))
    assert eff == 0


def test_no_price() -> None:
    assert effective_price(offer(listed_price=None)) == (None, [])
