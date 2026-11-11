"""Spec §4.3: code-side validation of extracted entities against the source text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal

from repricer.models import ExtractedOffer, OfferStatus, ReasonCode

THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
IMPLAUSIBLE_MULTIPLE = 100


@dataclass(frozen=True)
class ValidationResult:
    status: OfferStatus
    reject_reason: ReasonCode | None = None
    codes: list[ReasonCode] = field(default_factory=list)


def _amount_forms(d: Decimal) -> set[str]:
    two = f"{d:.2f}"
    forms = {two, two.replace(".", ",")}
    if d == d.to_integral_value():
        forms.add(str(int(d)))
    return forms


def amount_in_source(d: Decimal, raw_text: str) -> bool:
    text = THOUSANDS_RE.sub("", raw_text)
    haystacks = (raw_text, text)
    for form in _amount_forms(d):
        pat = re.compile(r"(?<![\d.])" + re.escape(form) + r"(?![\d])")
        if any(pat.search(h) for h in haystacks):
            return True
    return False


def percent_in_source(d: Decimal, raw_text: str) -> bool:
    forms = {f"{d:.2f}".rstrip("0").rstrip("."), str(int(d)) if d == int(d) else f"{d}"}
    return any(re.search(re.escape(f) + r"\s*%", raw_text) for f in forms)


def validate(
    offer: ExtractedOffer,
    raw_text: str,
    *,
    resolved_sku: str | None,
    current_price_cents: int | None,
) -> ValidationResult:
    if offer.listed_price is None:
        return ValidationResult(OfferStatus.REJECTED, ReasonCode.NO_PRICE_FOUND)
    if not amount_in_source(offer.listed_price, raw_text):
        return ValidationResult(OfferStatus.REJECTED, ReasonCode.PRICE_NOT_IN_SOURCE)
    if offer.coupon_amount is not None and not amount_in_source(offer.coupon_amount, raw_text):
        return ValidationResult(OfferStatus.REJECTED, ReasonCode.COUPON_NOT_IN_SOURCE)
    if offer.coupon_percent is not None and not percent_in_source(offer.coupon_percent, raw_text):
        return ValidationResult(OfferStatus.REJECTED, ReasonCode.COUPON_NOT_IN_SOURCE)
    if offer.listed_price == 0:
        return ValidationResult(OfferStatus.REJECTED, ReasonCode.ZERO_PRICE)
    if current_price_cents is not None and offer.listed_price * 100 > (
        current_price_cents * IMPLAUSIBLE_MULTIPLE
    ):
        return ValidationResult(OfferStatus.REJECTED, ReasonCode.IMPLAUSIBLE_PRICE)
    if resolved_sku is None:
        return ValidationResult(OfferStatus.REJECTED, ReasonCode.SKU_UNRESOLVED)

    codes: list[ReasonCode] = []
    if offer.currency != "USD":
        codes.append(ReasonCode.NON_USD)
    if offer.coupon_amount is not None and offer.coupon_percent is not None:
        return ValidationResult(
            OfferStatus.NEEDS_REVIEW,
            ReasonCode.AMBIGUOUS_COUPON,
            codes + [ReasonCode.AMBIGUOUS_COUPON],
        )
    if offer.is_bundle:
        return ValidationResult(
            OfferStatus.NEEDS_REVIEW, ReasonCode.BUNDLE_PRICE, codes + [ReasonCode.BUNDLE_PRICE]
        )
    return ValidationResult(OfferStatus.ACCEPTED, None, codes)
