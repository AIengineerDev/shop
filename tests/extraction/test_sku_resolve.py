from repricer.extraction.sku_resolve import resolve_sku

TITLES = [
    ("SKU-0001", "Acme Wireless Headphones X1"),
    ("SKU-0002", "Compact Blender C3"),
    ("SKU-0003", "Compact Blender D4"),
]


def test_scrape_sku_wins() -> None:
    assert (
        resolve_sku(scrape_sku="sku-0002", sku_hint=None, raw_text="x", titles=TITLES) == "SKU-0002"
    )


def test_hint_then_literal_then_title() -> None:
    assert (
        resolve_sku(scrape_sku=None, sku_hint="SKU-0001", raw_text="", titles=TITLES) == "SKU-0001"
    )
    assert (
        resolve_sku(scrape_sku=None, sku_hint=None, raw_text="[SKU-0003] blah", titles=TITLES)
        == "SKU-0003"
    )
    assert (
        resolve_sku(
            scrape_sku=None, sku_hint=None, raw_text="Acme Wireless Headphones X1 $9", titles=TITLES
        )
        == "SKU-0001"
    )


def test_ambiguous_or_weak_is_none() -> None:
    assert (
        resolve_sku(scrape_sku=None, sku_hint=None, raw_text="Compact Blender $9", titles=TITLES)
        is None
    )
    assert (
        resolve_sku(scrape_sku=None, sku_hint=None, raw_text="Headphones $9", titles=TITLES) is None
    )
    assert (
        resolve_sku(scrape_sku="NOPE", sku_hint="NOPE", raw_text="nothing", titles=TITLES) is None
    )
