"""Deterministic synthetic seed data.

Writes the *given* schema from the brief (REAL money columns) so ``migrate`` has real work
to do, and emits golden extraction fixtures whose expected values are known because the
generator built the text from templates.
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from repricer.clock import iso, utcnow

COMPETITORS = ("ShopMart", "DealHub", "PriceZone", "MegaBuy", "QuickCart")
ADJECTIVES = ("Wireless", "Compact", "Pro", "Ultra", "Classic", "Eco", "Smart", "Heavy-Duty")
NOUNS = (
    "Headphones",
    "Blender",
    "Backpack",
    "Desk Lamp",
    "Water Bottle",
    "Keyboard",
    "Mouse",
    "Yoga Mat",
    "Thermos",
    "Charger",
    "Speaker",
    "Tripod",
    "Kettle",
    "Monitor Stand",
)
PRICE_CLASSES = ("default", "default", "default", "clearance", "premium")

# The worked example from spec AC-R2. Always SKU-0001.
WORKED_SKU = "SKU-0001"
WORKED_TITLE = "Acme Wireless Headphones X1"


@dataclass
class SeedInventory:
    sku: str
    title: str
    cost_price: Decimal
    min_margin_percent: Decimal
    current_price: Decimal
    stock_count: int
    price_class: str


@dataclass
class SeedScrape:
    scrape_id: int
    sku: str | None
    competitor: str
    raw_text: str
    scraped_at: datetime
    expected: dict[str, Any]  # ExtractedOffer fields minus confidence/scrape_id
    expected_status: str  # accepted | rejected | needs_review
    expected_reason: str | None
    case: str


@dataclass
class SeedData:
    inventory: list[SeedInventory]
    scrapes: list[SeedScrape]
    meta: dict[str, Any] = field(default_factory=dict)


def _money(rng: random.Random, lo: int, hi: int) -> Decimal:
    cents = rng.randint(lo * 100, hi * 100)
    return (Decimal(cents) / 100).quantize(Decimal("0.01"))


def _d(x: Decimal | int | str) -> str:
    return str(Decimal(x).quantize(Decimal("0.01")))


TemplateResult = tuple[str, dict[str, Any], str, str | None]
Template = Callable[[random.Random, SeedInventory, Decimal], TemplateResult]


def _base(price: Decimal | None, **kw: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "sku_hint": None,
        "listed_price": None if price is None else _d(price),
        "currency": "USD" if price is not None else "UNKNOWN",
        "coupon_amount": None,
        "coupon_percent": None,
        "shipping": None,
        "condition": "unknown",
        "in_stock": None,
        "is_bundle": False,
    }
    d.update(kw)
    return d


def t_plain(rng: random.Random, inv: SeedInventory, p: Decimal) -> TemplateResult:
    return f"{inv.title} - ${_d(p)}", _base(p), "accepted", None


def t_coupon_amount(rng: random.Random, inv: SeedInventory, p: Decimal) -> TemplateResult:
    c = Decimal(rng.choice([3, 5, 10]))
    return (
        f"{inv.title} on sale for ${_d(p)} with ${int(c)} coupon",
        _base(p, coupon_amount=_d(c)),
        "accepted",
        None,
    )


def t_coupon_percent(rng: random.Random, inv: SeedInventory, p: Decimal) -> TemplateResult:
    pct = rng.choice([10, 15, 20])
    return (
        f"{inv.title} ${_d(p)} - take {pct}% off at checkout with code SAVE{pct}",
        _base(p, coupon_percent=_d(pct)),
        "accepted",
        None,
    )


def t_both_coupons(rng: random.Random, inv: SeedInventory, p: Decimal) -> TemplateResult:
    return (
        f"{inv.title} ${_d(p)}: $5 off OR 10% off, cannot combine",
        _base(p, coupon_amount="5.00", coupon_percent="10.00"),
        "needs_review",
        "AMBIGUOUS_COUPON",
    )


def t_free_ship(rng: random.Random, inv: SeedInventory, p: Decimal) -> TemplateResult:
    return f"{inv.title} ${_d(p)} FREE SHIPPING", _base(p, shipping="0.00"), "accepted", None


def t_paid_ship(rng: random.Random, inv: SeedInventory, p: Decimal) -> TemplateResult:
    s = Decimal(rng.choice(["4.99", "6.95", "8.00"]))
    return (
        f"{inv.title} ${_d(p)} + ${_d(s)} shipping",
        _base(p, shipping=_d(s)),
        "accepted",
        None,
    )


def t_was_now(rng: random.Random, inv: SeedInventory, p: Decimal) -> TemplateResult:
    old = p + Decimal(rng.randint(5, 20))
    return (
        f"{inv.title}: was ${_d(old)}, now ${_d(p)}",
        _base(p),
        "accepted",
        None,
    )


def t_bundle(rng: random.Random, inv: SeedInventory, p: Decimal) -> TemplateResult:
    bundle = (p * 2 - Decimal("5")).quantize(Decimal("0.01"))
    return (
        f"{inv.title} 2 for ${_d(bundle)}",
        _base(bundle, is_bundle=True),
        "needs_review",
        "BUNDLE_PRICE",
    )


def t_non_usd(rng: random.Random, inv: SeedInventory, p: Decimal) -> TemplateResult:
    cur, sym = rng.choice([("CAD", "CA$"), ("EUR", "€"), ("GBP", "£")])
    return (
        f"{inv.title} {sym}{_d(p)}",
        _base(p, currency=cur),
        "accepted",
        "NON_USD",
    )


def t_per_unit(rng: random.Random, inv: SeedInventory, p: Decimal) -> TemplateResult:
    return f"{inv.title} ${_d(p)} each, in stock", _base(p, in_stock=True), "accepted", None


def t_typo_symbol(rng: random.Random, inv: SeedInventory, p: Decimal) -> TemplateResult:
    variant = rng.choice(["$ {p}", "USD {p}", "{p} USD", "US${p}"])
    return (
        f"{inv.title} {variant.format(p=_d(p))}",
        _base(p),
        "accepted",
        None,
    )


def t_no_price(rng: random.Random, inv: SeedInventory, p: Decimal) -> TemplateResult:
    return (
        f"{inv.title} - currently unavailable, check back soon",
        _base(None, in_stock=False),
        "rejected",
        "NO_PRICE_FOUND",
    )


def t_used(rng: random.Random, inv: SeedInventory, p: Decimal) -> TemplateResult:
    return (
        f"{inv.title} (used - good) ${_d(p)}",
        _base(p, condition="used"),
        "accepted",
        None,
    )


def t_refurb(rng: random.Random, inv: SeedInventory, p: Decimal) -> TemplateResult:
    return (
        f"Refurbished {inv.title} ${_d(p)}",
        _base(p, condition="refurbished"),
        "accepted",
        None,
    )


def t_new_explicit(rng: random.Random, inv: SeedInventory, p: Decimal) -> TemplateResult:
    return (
        f"Brand new {inv.title} ${_d(p)} ships free",
        _base(p, condition="new", shipping="0.00"),
        "accepted",
        None,
    )


def t_sku_in_text(rng: random.Random, inv: SeedInventory, p: Decimal) -> TemplateResult:
    # scrape.sku is None for this template; SKU must be resolved from the text.
    return (
        f"[{inv.sku}] {inv.title} ${_d(p)}",
        _base(p, sku_hint=inv.sku),
        "accepted",
        None,
    )


def t_thousands(rng: random.Random, inv: SeedInventory, p: Decimal) -> TemplateResult:
    big = p + Decimal(1000)
    return f"{inv.title} ${big:,.2f}", _base(big), "accepted", None


TEMPLATES: list[tuple[str, Template, int]] = [
    ("plain", t_plain, 40),
    ("coupon_amount", t_coupon_amount, 25),
    ("coupon_percent", t_coupon_percent, 20),
    ("both_coupons", t_both_coupons, 8),
    ("free_ship", t_free_ship, 18),
    ("paid_ship", t_paid_ship, 15),
    ("was_now", t_was_now, 15),
    ("bundle", t_bundle, 8),
    ("non_usd", t_non_usd, 8),
    ("per_unit", t_per_unit, 8),
    ("typo_symbol", t_typo_symbol, 10),
    ("no_price", t_no_price, 6),
    ("used", t_used, 6),
    ("refurb", t_refurb, 4),
    ("new_explicit", t_new_explicit, 10),
    ("sku_in_text", t_sku_in_text, 8),
    ("thousands", t_thousands, 4),
]


def generate(
    seed: int = 42, n_skus: int = 50, n_scrapes: int = 200, now: datetime | None = None
) -> SeedData:
    rng = random.Random(seed)
    now = now or utcnow()
    inventory: list[SeedInventory] = [
        SeedInventory(
            sku=WORKED_SKU,
            title=WORKED_TITLE,
            cost_price=Decimal("30.00"),
            min_margin_percent=Decimal("20.00"),
            current_price=Decimal("45.00"),
            stock_count=25,
            price_class="default",
        )
    ]
    used_titles = {WORKED_TITLE}
    for i in range(2, n_skus + 1):
        while True:
            suffix = f"{rng.choice('ABCDEFGH')}{rng.randint(1, 9)}"
            title = f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)} {suffix}"
            if title not in used_titles:
                used_titles.add(title)
                break
        cost = _money(rng, 8, 120)
        margin = Decimal(rng.choice([10, 15, 20, 25, 30]))
        current = (
            cost * (1 + margin / 100) * Decimal(rng.choice(["1.05", "1.15", "1.25", "1.4"]))
        ).quantize(Decimal("0.01"))
        inventory.append(
            SeedInventory(
                sku=f"SKU-{i:04d}",
                title=title,
                cost_price=cost,
                min_margin_percent=margin,
                current_price=current,
                stock_count=rng.choice([0, 1, 2, 3, 5, 12, 40, 150]),
                price_class=rng.choice(PRICE_CLASSES),
            )
        )

    by_sku = {inv.sku: inv for inv in inventory}
    scrapes: list[SeedScrape] = []

    # Worked example first: AC-R2.
    worked = by_sku[WORKED_SKU]
    scrapes.append(
        SeedScrape(
            scrape_id=1,
            sku=WORKED_SKU,
            competitor="DealHub",
            raw_text=f"{WORKED_TITLE} on sale for $42.99 with $5 coupon",
            scraped_at=now - timedelta(hours=2),
            expected=_base(Decimal("42.99"), coupon_amount="5.00"),
            expected_status="accepted",
            expected_reason=None,
            case="coupon_amount",
        )
    )
    # Make the worked example's offer eligible even for 'new'-only policies.
    scrapes[0].raw_text = f"Brand new {WORKED_TITLE} on sale for $42.99 with $5 coupon"
    scrapes[0].expected["condition"] = "new"
    _ = worked

    weights = [w for _, _, w in TEMPLATES]
    sid = 2
    while len(scrapes) < n_scrapes:
        inv = rng.choice(inventory)
        name, fn, _w = rng.choices(TEMPLATES, weights=weights, k=1)[0]
        # competitor price: around current price, sometimes below, sometimes above
        p = (
            inv.current_price * Decimal(rng.choice(["0.85", "0.9", "0.95", "1.0", "1.05", "1.1"]))
        ).quantize(Decimal("0.01"))
        raw, expected, status, reason = fn(rng, inv, p)
        scrapes.append(
            SeedScrape(
                scrape_id=sid,
                sku=None if name == "sku_in_text" else inv.sku,
                competitor=rng.choice(COMPETITORS),
                raw_text=raw,
                scraped_at=now - timedelta(hours=rng.randint(0, 60), minutes=rng.randint(0, 59)),
                expected=expected,
                expected_status=status,
                expected_reason=reason,
                case=name,
            )
        )
        sid += 1

    return SeedData(inventory, scrapes, {"seed": seed, "generated_at": iso(now)})


LEGACY_DDL = """
CREATE TABLE inventory (
    SKU TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    cost_price REAL NOT NULL,
    min_margin_percent REAL NOT NULL,
    current_price REAL NOT NULL,
    stock_count INTEGER NOT NULL,
    price_class TEXT
);
CREATE TABLE competitor_scrapes (
    scrape_id INTEGER PRIMARY KEY,
    sku TEXT,
    competitor TEXT,
    raw_text TEXT NOT NULL,
    scraped_at TEXT
);
"""


def write_legacy_db(conn: sqlite3.Connection, data: SeedData) -> None:
    """Write the brief's schema (REAL money). ``migrate`` converts it."""
    conn.executescript(LEGACY_DDL)
    conn.executemany(
        "INSERT INTO inventory VALUES (?,?,?,?,?,?,?)",
        [
            (
                i.sku,
                i.title,
                float(i.cost_price),
                float(i.min_margin_percent),
                float(i.current_price),
                i.stock_count,
                i.price_class,
            )
            for i in data.inventory
        ],
    )
    conn.executemany(
        "INSERT INTO competitor_scrapes VALUES (?,?,?,?,?)",
        [(s.scrape_id, s.sku, s.competitor, s.raw_text, iso(s.scraped_at)) for s in data.scrapes],
    )


def write_fixtures(directory: Path, data: SeedData) -> int:
    """One JSON per template case, each with a list of scrapes. Returns scrape count."""
    directory.mkdir(parents=True, exist_ok=True)
    by_case: dict[str, list[dict[str, Any]]] = {}
    for s in data.scrapes:
        by_case.setdefault(s.case, []).append(
            {
                "scrape_id": s.scrape_id,
                "sku": s.sku,
                "competitor": s.competitor,
                "raw_text": s.raw_text,
                "expected": s.expected,
                "expected_status": s.expected_status,
                "expected_reason": s.expected_reason,
            }
        )
    for case, items in sorted(by_case.items()):
        (directory / f"{case}.json").write_text(json.dumps(items, indent=2) + "\n")
    return len(data.scrapes)
