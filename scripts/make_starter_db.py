#!/usr/bin/env python
"""Build a synthetic starter SQLite DB with canonical_categories and incoming_supplier_feeds.

Usage: python scripts/make_starter_db.py --out data/starter.sqlite [--seed 7] [--rows 200]
Deterministic for a given seed. Includes the exact sample row from the spec (feed_item_id = SAMPLE-0001).
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SAMPLE_FEED_ITEM_ID = "SAMPLE-0001"
SAMPLE_TITLE = "Men's Dry-Fit Athletic Gym Shorts w/ Pockets Pack of 2 (Black/Navy, L) - Running Workout"

SIZES = ["XS", "S", "M", "L", "XL", "XXL"]
COLORS = ["black", "navy", "white", "gray", "red", "blue", "green", "yellow", "pink", "brown", "orange", "purple"]
GENDERS = ["men", "women", "unisex", "kids"]
MATERIALS_APPAREL = ["cotton", "polyester", "nylon", "spandex blend", "merino wool", "fleece"]
MATERIALS_KITCHEN = ["stainless steel", "cast iron", "ceramic", "glass", "silicone", "bamboo"]


def spec(key: str, type_: str, **kw: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"key": key, "type": type_}
    d.update(kw)
    return d


APPAREL_ATTRS = [
    spec("gender", "enum", allowed=GENDERS),
    spec("size", "enum", allowed=SIZES, multi=True),
    spec("color", "enum", allowed=COLORS + ["multicolor"], multi=True),
    spec("pack_count", "int", min=1, max=24),
    spec("material", "string", max_len=64),
]
FOOTWEAR_ATTRS = [
    spec("gender", "enum", allowed=GENDERS),
    spec("size", "enum", allowed=[str(s) for s in range(5, 15)], multi=True),
    spec("color", "enum", allowed=COLORS + ["multicolor"], multi=True),
    spec("material", "string", max_len=64),
]
KITCHEN_ATTRS = [
    spec("material", "string", max_len=64),
    spec("pack_count", "int", min=1, max=48),
    spec("color", "enum", allowed=COLORS + ["silver", "multicolor"], multi=True),
]
DRINKWARE_ATTRS = [
    spec("material", "string", max_len=64),
    spec("capacity_oz", "float", min=1, max=128),
    spec("color", "enum", allowed=COLORS + ["silver", "multicolor"], multi=True),
]
SPORT_ATTRS = [
    spec("color", "enum", allowed=COLORS + ["multicolor"], multi=True),
    spec("pack_count", "int", min=1, max=12),
    spec("material", "string", max_len=64),
]
ELECTRONICS_ATTRS = [
    spec("color", "enum", allowed=COLORS + ["silver", "multicolor"], multi=True),
    spec("pack_count", "int", min=1, max=10),
    spec("connector", "enum", allowed=["usb-a", "usb-c", "lightning", "micro-usb", "3.5mm", "none"]),
]
OUTDOOR_ATTRS = [
    spec("color", "enum", allowed=COLORS + ["multicolor"], multi=True),
    spec("capacity_liters", "float", min=1, max=120),
    spec("material", "string", max_len=64),
]

# (category_id, name, parent_id, attrs or None, prohibited)
CATEGORIES: list[tuple[str, str, str | None, list[dict[str, Any]] | None, str]] = [
    ("apparel", "Apparel", None, None, "counterfeit,replica"),
    ("apparel.men", "Men's Clothing", "apparel", None, ""),
    ("apparel.men.tops", "Men's Tops", "apparel.men", None, ""),
    ("apparel.men.tops.tshirts", "T-Shirts", "apparel.men.tops", APPAREL_ATTRS, ""),
    ("apparel.men.tops.hoodies", "Hoodies & Sweatshirts", "apparel.men.tops", APPAREL_ATTRS, ""),
    ("apparel.men.tops.polos", "Polo Shirts", "apparel.men.tops", APPAREL_ATTRS, ""),
    ("apparel.men.bottoms", "Men's Bottoms", "apparel.men", None, ""),
    ("apparel.men.bottoms.shorts", "Shorts", "apparel.men.bottoms", APPAREL_ATTRS, ""),
    ("apparel.men.bottoms.jeans", "Jeans", "apparel.men.bottoms", APPAREL_ATTRS, ""),
    ("apparel.men.bottoms.joggers", "Joggers & Sweatpants", "apparel.men.bottoms", APPAREL_ATTRS, ""),
    ("apparel.women", "Women's Clothing", "apparel", None, ""),
    ("apparel.women.tops", "Women's Tops", "apparel.women", None, ""),
    ("apparel.women.tops.tshirts", "T-Shirts", "apparel.women.tops", APPAREL_ATTRS, ""),
    ("apparel.women.tops.blouses", "Blouses", "apparel.women.tops", APPAREL_ATTRS, ""),
    ("apparel.women.bottoms", "Women's Bottoms", "apparel.women", None, ""),
    ("apparel.women.bottoms.leggings", "Leggings", "apparel.women.bottoms", APPAREL_ATTRS, ""),
    ("apparel.women.bottoms.skirts", "Skirts", "apparel.women.bottoms", APPAREL_ATTRS, ""),
    ("apparel.women.dresses", "Dresses", "apparel.women", APPAREL_ATTRS, ""),
    ("apparel.kids", "Kids' Clothing", "apparel", None, "choking hazard"),
    ("apparel.kids.tshirts", "Kids' T-Shirts", "apparel.kids", APPAREL_ATTRS, ""),
    ("apparel.kids.pajamas", "Kids' Pajamas", "apparel.kids", APPAREL_ATTRS, "flammable"),
    ("footwear", "Footwear", None, None, "counterfeit,replica"),
    ("footwear.running", "Running Shoes", "footwear", FOOTWEAR_ATTRS, ""),
    ("footwear.sandals", "Sandals", "footwear", FOOTWEAR_ATTRS, ""),
    ("footwear.boots", "Boots", "footwear", FOOTWEAR_ATTRS, ""),
    ("kitchen", "Kitchen & Dining", None, None, ""),
    ("kitchen.cookware", "Cookware", "kitchen", None, "teflon flakes"),
    ("kitchen.cookware.pans", "Frying Pans & Skillets", "kitchen.cookware", KITCHEN_ATTRS, ""),
    ("kitchen.cookware.pots", "Pots & Dutch Ovens", "kitchen.cookware", KITCHEN_ATTRS, ""),
    ("kitchen.cutlery", "Knives & Cutlery", "kitchen", None, "switchblade,butterfly knife"),
    ("kitchen.cutlery.chef_knives", "Chef Knives", "kitchen.cutlery", KITCHEN_ATTRS, ""),
    ("kitchen.cutlery.knife_sets", "Knife Sets", "kitchen.cutlery", KITCHEN_ATTRS, ""),
    ("kitchen.drinkware", "Drinkware", "kitchen", None, ""),
    ("kitchen.drinkware.bottles", "Water Bottles", "kitchen.drinkware", DRINKWARE_ATTRS, "bpa"),
    ("kitchen.drinkware.mugs", "Mugs & Tumblers", "kitchen.drinkware", DRINKWARE_ATTRS, ""),
    ("kitchen.storage", "Food Storage", "kitchen", KITCHEN_ATTRS, ""),
    ("sport", "Sporting Goods", None, None, "steroid,performance enhancing"),
    ("sport.fitness", "Fitness", "sport", None, ""),
    ("sport.fitness.yoga_mats", "Yoga Mats", "sport.fitness", SPORT_ATTRS, ""),
    ("sport.fitness.resistance_bands", "Resistance Bands", "sport.fitness", SPORT_ATTRS, ""),
    ("sport.fitness.dumbbells", "Dumbbells & Weights", "sport.fitness", SPORT_ATTRS, ""),
    ("sport.outdoor", "Outdoor & Camping", "sport", None, "bear spray,pepper spray"),
    ("sport.outdoor.backpacks", "Hiking Backpacks", "sport.outdoor", OUTDOOR_ATTRS, ""),
    ("sport.outdoor.tents", "Tents", "sport.outdoor", OUTDOOR_ATTRS, ""),
    ("sport.outdoor.coolers", "Coolers", "sport.outdoor", OUTDOOR_ATTRS, ""),
    ("sport.ball", "Ball Sports", "sport", SPORT_ATTRS, ""),
    ("electronics", "Electronics Accessories", None, None, "jammer,signal blocker"),
    ("electronics.cables", "Cables & Chargers", "electronics", ELECTRONICS_ATTRS, ""),
    ("electronics.audio", "Headphones & Earbuds", "electronics", ELECTRONICS_ATTRS, ""),
    ("electronics.cases", "Phone Cases", "electronics", ELECTRONICS_ATTRS, ""),
    ("electronics.power", "Power Banks", "electronics", ELECTRONICS_ATTRS, "lithium leak"),
]

SUPPLIERS = ["sup_acme", "sup_zenith", "sup_orbit", "sup_nimbus"]

# leaf -> list of (noun phrases, extra descriptors, bullets, attr kind)
PRODUCT_TEMPLATES: dict[str, dict[str, Any]] = {
    "apparel.men.tops.tshirts": {
        "nouns": ["Crew Neck T-Shirt", "Graphic Tee", "V-Neck Tee"],
        "kind": "apparel",
        "gender": "men",
    },
    "apparel.men.tops.hoodies": {
        "nouns": ["Pullover Hoodie", "Zip-Up Hooded Sweatshirt", "Fleece Hoodie"],
        "kind": "apparel",
        "gender": "men",
    },
    "apparel.men.tops.polos": {"nouns": ["Pique Polo Shirt", "Golf Polo"], "kind": "apparel", "gender": "men"},
    "apparel.men.bottoms.shorts": {
        "nouns": ["Athletic Gym Shorts", "Cargo Shorts", "Swim Trunks Shorts"],
        "kind": "apparel",
        "gender": "men",
    },
    "apparel.men.bottoms.jeans": {
        "nouns": ["Slim Fit Jeans", "Relaxed Denim Jeans"],
        "kind": "apparel",
        "gender": "men",
    },
    "apparel.men.bottoms.joggers": {
        "nouns": ["Jogger Sweatpants", "Tapered Joggers"],
        "kind": "apparel",
        "gender": "men",
    },
    "apparel.women.tops.tshirts": {"nouns": ["Scoop Neck T-Shirt", "Fitted Tee"], "kind": "apparel", "gender": "women"},
    "apparel.women.tops.blouses": {
        "nouns": ["Chiffon Blouse", "Button-Down Blouse"],
        "kind": "apparel",
        "gender": "women",
    },
    "apparel.women.bottoms.leggings": {
        "nouns": ["High Waist Leggings", "Yoga Leggings"],
        "kind": "apparel",
        "gender": "women",
    },
    "apparel.women.bottoms.skirts": {
        "nouns": ["Pleated Midi Skirt", "A-Line Skirt"],
        "kind": "apparel",
        "gender": "women",
    },
    "apparel.women.dresses": {"nouns": ["Wrap Dress", "Maxi Sundress"], "kind": "apparel", "gender": "women"},
    "apparel.kids.tshirts": {"nouns": ["Kids Graphic T-Shirt", "Toddler Tee"], "kind": "apparel", "gender": "kids"},
    "apparel.kids.pajamas": {"nouns": ["Kids Pajama Set", "Toddler Sleepwear"], "kind": "apparel", "gender": "kids"},
    "footwear.running": {"nouns": ["Running Shoes", "Trail Running Sneakers"], "kind": "footwear"},
    "footwear.sandals": {"nouns": ["Slide Sandals", "Sport Sandals"], "kind": "footwear"},
    "footwear.boots": {"nouns": ["Waterproof Hiking Boots", "Chelsea Boots"], "kind": "footwear"},
    "kitchen.cookware.pans": {"nouns": ["Nonstick Frying Pan", "Cast Iron Skillet"], "kind": "kitchen"},
    "kitchen.cookware.pots": {"nouns": ["Stock Pot with Lid", "Enameled Dutch Oven"], "kind": "kitchen"},
    "kitchen.cutlery.chef_knives": {"nouns": ["8 Inch Chef Knife", "Santoku Chef Knife"], "kind": "kitchen"},
    "kitchen.cutlery.knife_sets": {"nouns": ["Kitchen Knife Set with Block", "Steak Knife Set"], "kind": "kitchen"},
    "kitchen.drinkware.bottles": {"nouns": ["Insulated Water Bottle", "Sports Water Bottle"], "kind": "drinkware"},
    "kitchen.drinkware.mugs": {"nouns": ["Travel Tumbler Mug", "Ceramic Coffee Mug"], "kind": "drinkware"},
    "kitchen.storage": {"nouns": ["Food Storage Container Set", "Glass Meal Prep Containers"], "kind": "kitchen"},
    "sport.fitness.yoga_mats": {"nouns": ["Non-Slip Yoga Mat", "Thick Exercise Yoga Mat"], "kind": "sport"},
    "sport.fitness.resistance_bands": {"nouns": ["Resistance Bands Set", "Loop Resistance Band"], "kind": "sport"},
    "sport.fitness.dumbbells": {"nouns": ["Adjustable Dumbbell", "Neoprene Dumbbells Weights"], "kind": "sport"},
    "sport.outdoor.backpacks": {"nouns": ["Hiking Backpack", "Daypack Backpack"], "kind": "outdoor"},
    "sport.outdoor.tents": {"nouns": ["Camping Tent", "Backpacking Tent"], "kind": "outdoor"},
    "sport.outdoor.coolers": {"nouns": ["Hard Cooler", "Soft Sided Cooler Bag"], "kind": "outdoor"},
    "sport.ball": {"nouns": ["Soccer Ball", "Basketball"], "kind": "sport"},
    "electronics.cables": {"nouns": ["Fast Charging Cable", "Braided Charger Cable"], "kind": "electronics"},
    "electronics.audio": {"nouns": ["Wireless Earbuds", "Over-Ear Headphones"], "kind": "electronics"},
    "electronics.cases": {"nouns": ["Shockproof Phone Case", "Clear Phone Case"], "kind": "electronics"},
    "electronics.power": {"nouns": ["Portable Power Bank", "Power Bank Charger"], "kind": "electronics"},
}

TAILS = ["Running Workout", "Gift Idea", "Travel Camping", "Home Office", "Gym Training", "Everyday Casual", ""]
DESCRIPTORS = ["Premium", "Lightweight", "Heavy Duty", "Breathable", "Quick Dry", "Dry-Fit", "Eco Friendly", ""]
BULLET_SEPS = ["\n", " • ", "; ", "\n• "]


def title_case_word(w: str) -> str:
    return w[:1].upper() + w[1:]


def make_row(rng: random.Random, idx: int, leaf: str, supplier: str, base_time: datetime) -> dict[str, Any]:
    tpl = PRODUCT_TEMPLATES[leaf]
    kind = tpl["kind"]
    noun = rng.choice(tpl["nouns"])
    desc = rng.choice(DESCRIPTORS)
    tail = rng.choice(TAILS)
    colors = rng.sample(COLORS, rng.choice([1, 1, 2, 3]))
    pack = rng.choice([1, 1, 1, 2, 3, 4, 6])
    tags: dict[str, str] = {}
    bullets: list[str] = []

    gender = tpl.get("gender")
    gender_prefix = ""
    if kind == "apparel":
        gender_prefix = {
            "men": rng.choice(["Men's", "Mens", "Men"]),
            "women": rng.choice(["Women's", "Womens", "Ladies"]),
            "kids": rng.choice(["Kids'", "Kids", "Boys"]),
        }[gender]
        sizes = rng.sample(SIZES, rng.choice([1, 1, 2]))
        material = rng.choice(MATERIALS_APPAREL)
        tags["size" if supplier != "sup_acme" else "sze"] = "/".join(rng.choice([s, s.lower()]) for s in sizes)
        bullets.append(f"Material: {material}")
        bullets.append("Machine washable")
        var = f"({'/'.join(title_case_word(c) for c in colors)}, {'/'.join(sizes)})"
    elif kind == "footwear":
        gender = rng.choice(["men", "women", "unisex"])
        gender_prefix = {"men": "Men's", "women": "Women's", "unisex": "Unisex"}[gender]
        sizes = [str(rng.randint(6, 13))]
        material = rng.choice(["mesh", "leather", "synthetic", "suede"])
        tags["shoe_size"] = sizes[0]
        bullets.append(f"Upper: {material}")
        bullets.append("Cushioned sole")
        var = f"({title_case_word(colors[0])}, Size {sizes[0]})"
    elif kind in ("kitchen", "drinkware"):
        material = rng.choice(MATERIALS_KITCHEN)
        bullets.append(f"Made of {material}")
        bullets.append("Dishwasher safe")
        var = f"({title_case_word(colors[0])})"
        if kind == "drinkware":
            oz = rng.choice([12, 16, 20, 24, 32, 40])
            tags["volume_oz"] = str(oz)
            noun = f"{oz} oz {noun}"
    elif kind == "sport":
        material = rng.choice(["TPE foam", "latex", "neoprene", "rubber"])
        bullets.append(f"Material: {material}")
        bullets.append("Non-slip surface")
        var = f"({title_case_word(colors[0])})"
    elif kind == "outdoor":
        material = rng.choice(["ripstop nylon", "polyester", "canvas"])
        liters = rng.choice([20, 30, 40, 50, 65])
        tags["capacity_l"] = str(liters)
        bullets.append(f"Material: {material}")
        bullets.append(f"{liters}L capacity")
        var = f"({title_case_word(colors[0])})"
    else:  # electronics
        material = ""
        connector = rng.choice(["usb-c", "usb-a", "lightning", "micro-usb", "3.5mm"])
        tags["connector"] = connector.upper() if rng.random() < 0.5 else connector
        bullets.append(f"Connector: {connector}")
        bullets.append("12-month warranty")
        var = f"({title_case_word(colors[0])})"

    color_key = {"sup_acme": "clr", "sup_zenith": "kolor"}.get(supplier, rng.choice(["colour", "color"]))
    tags[color_key] = ",".join(colors)
    if kind == "apparel":
        tags[rng.choice(["gender", "sex"])] = {
            "men": rng.choice(["M", "men", "male"]),
            "women": rng.choice(["F", "women"]),
            "kids": rng.choice(["K", "kids"]),
        }[gender]
    if pack > 1:
        tags["qty"] = str(pack)

    pack_phrase = rng.choice([f"Pack of {pack}", f"{pack}-Pack", f"{pack}pk", f"Set of {pack}"]) if pack > 1 else ""
    pockets = rng.choice(["w/ Pockets", "with Pockets", ""]) if kind == "apparel" else ""
    parts = [gender_prefix, desc, noun, pockets, pack_phrase, var]
    title = " ".join(p for p in parts if p)
    if tail:
        title += f" - {tail}"
    if rng.random() < 0.15:
        title = title.replace("  ", " ").upper() if rng.random() < 0.3 else title + "  "

    sep = rng.choice(BULLET_SEPS)
    if rng.random() < 0.2:
        bullets.append(bullets[0])  # duplicate bullet
    if rng.random() < 0.2:
        bullets.append("   ")  # blank bullet
    raw_bullets = sep.join(bullets)

    price = round(rng.uniform(6, 180), 2)
    raw_price = rng.choice([f"${price}", f"{price}", f"USD {price}", f"${price} - ${round(price * 1.3, 2)}"])
    tag_sep = rng.choice([";", "|", ","])
    raw_tags = tag_sep.join(f"{k}={v}" for k, v in tags.items())
    received = (base_time + timedelta(minutes=idx * 7)).isoformat(timespec="seconds")
    return {
        "feed_item_id": f"{supplier.upper()}-{idx:05d}",
        "supplier_id": supplier,
        "raw_title": title,
        "raw_bullets": raw_bullets,
        "raw_tags": raw_tags,
        "raw_price": raw_price,
        "received_at": received,
        "_expected_leaf": leaf,
    }


def inject_bad_rows(rng: random.Random, rows: list[dict[str, Any]]) -> None:
    """Mutate ~10% of rows into deliberately problematic listings."""
    n = max(1, len(rows) // 10)
    victims = rng.sample(range(1, len(rows)), n)  # never touch index 0 (the sample row)
    kinds = ["empty_title", "prohibited_global", "prohibited_category", "missing_size", "bad_price", "nonleaf_bait"]
    for i, v in enumerate(victims):
        row = rows[v]
        kind = kinds[i % len(kinds)]
        row["_bad"] = kind
        if kind == "empty_title":
            row["raw_title"] = rng.choice(["", "   ", None])
        elif kind == "prohibited_global":
            row["raw_title"] = f"{row['raw_title']} Replica"
        elif kind == "prohibited_category":
            leaf = row["_expected_leaf"]
            if leaf.startswith("kitchen.cutlery"):
                row["raw_bullets"] += "\nAlso works as a butterfly knife"
            elif leaf.startswith("sport.outdoor"):
                row["raw_bullets"] += "\nIncludes bear spray holster"
            elif leaf.startswith("apparel.kids"):
                row["raw_bullets"] += "\nSmall parts, choking hazard"
            elif leaf.startswith("electronics"):
                row["raw_bullets"] += "\nDoubles as a signal blocker"
            else:
                row["raw_title"] = f"Counterfeit {row['raw_title']}"
        elif kind == "missing_size":
            row["raw_tags"] = ";".join(
                t for t in row["raw_tags"].split(";") if not t.lower().startswith(("size", "sze", "shoe_size"))
            )
            row["raw_title"] = row["raw_title"].replace(", L)", ")").replace(", M)", ")").replace(", S)", ")")
            row["raw_title"] = row["raw_title"].replace(", XL)", ")").replace(", XS)", ")").replace(", XXL)", ")")
        elif kind == "bad_price":
            row["raw_price"] = rng.choice(["N/A", "call for price", "", "$0"])
        elif kind == "nonleaf_bait":
            row["raw_title"] = "Men's Clothing Assorted Bundle Mystery Box"
            row["raw_bullets"] = "Assorted apparel items"


def build(out: Path, seed: int, n_rows: int) -> None:
    rng = random.Random(seed)
    base_time = datetime(2026, 11, 1, 8, 0, tzinfo=UTC)
    leaves = list(PRODUCT_TEMPLATES)

    rows: list[dict[str, Any]] = []
    sample = {
        "feed_item_id": SAMPLE_FEED_ITEM_ID,
        "supplier_id": "sup_acme",
        "raw_title": SAMPLE_TITLE,
        "raw_bullets": (
            "Moisture wicking fabric\n• Two side pockets\n• Elastic waistband with drawstring\n•    \n"
            "• Moisture wicking fabric"
        ),
        "raw_tags": "sze=L;clr=blk,nvy;qty=2;material=polyester;sex=M",
        "raw_price": "$24.99",
        "received_at": base_time.isoformat(timespec="seconds"),
        "_expected_leaf": "apparel.men.bottoms.shorts",
    }
    rows.append(sample)
    for idx in range(1, n_rows):
        leaf = leaves[idx % len(leaves)] if idx < len(leaves) * 2 else rng.choice(leaves)
        supplier = rng.choice(SUPPLIERS)
        rows.append(make_row(rng, idx, leaf, supplier, base_time))
    inject_bad_rows(rng, rows)

    if out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(out))
    conn.executescript(
        """
        CREATE TABLE canonical_categories (
            category_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            parent_id TEXT,
            required_attributes_json TEXT,
            prohibited_keywords TEXT
        );
        CREATE TABLE incoming_supplier_feeds (
            feed_item_id TEXT PRIMARY KEY,
            supplier_id TEXT NOT NULL,
            raw_title TEXT,
            raw_bullets TEXT,
            raw_tags TEXT,
            raw_price TEXT,
            received_at TEXT,
            expected_leaf TEXT,
            injected_defect TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO canonical_categories VALUES (?,?,?,?,?)",
        [
            (cid, name, parent, None if attrs is None else json.dumps(attrs), prohibited)
            for cid, name, parent, attrs, prohibited in CATEGORIES
        ],
    )
    conn.executemany(
        "INSERT INTO incoming_supplier_feeds VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (
                r["feed_item_id"],
                r["supplier_id"],
                r["raw_title"],
                r["raw_bullets"],
                r["raw_tags"],
                r["raw_price"],
                r["received_at"],
                r["_expected_leaf"],
                r.get("_bad"),
            )
            for r in rows
        ],
    )
    conn.commit()
    conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/starter.sqlite"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rows", type=int, default=200)
    args = ap.parse_args()
    build(args.out, args.seed, args.rows)
    print(f"wrote {args.out} ({len(CATEGORIES)} categories, {args.rows} feed rows, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
