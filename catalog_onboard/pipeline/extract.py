"""Deterministic extractors: regexes and dictionaries only. No LLM, no I/O."""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

from catalog_onboard.models import AttrValue
from catalog_onboard.rules import Rules

APPAREL_SIZES = ("XS", "S", "M", "L", "XL", "XXL", "XXXL")
SIZE_ALIASES = {
    "x-small": "XS",
    "xsmall": "XS",
    "small": "S",
    "sm": "S",
    "medium": "M",
    "md": "M",
    "med": "M",
    "large": "L",
    "lg": "L",
    "x-large": "XL",
    "xlarge": "XL",
    "extra large": "XL",
    "xx-large": "XXL",
    "xxlarge": "XXL",
    "2xl": "XXL",
    "3xl": "XXXL",
}
COLOR_WORDS = frozenset(
    {
        "black",
        "navy",
        "white",
        "gray",
        "red",
        "blue",
        "green",
        "yellow",
        "pink",
        "brown",
        "orange",
        "purple",
        "silver",
        "gold",
        "beige",
        "teal",
        "multicolor",
        "khaki",
        "olive",
        "maroon",
        "charcoal",
    }
)
COLOR_ALIASES = {"grey": "gray", "multi": "multicolor", "multi-color": "multicolor", "multicolour": "multicolor"}
GENDER_WORDS = {"men": "men", "women": "women", "kids": "kids", "unisex": "unisex", "boys": "kids", "girls": "kids"}
NUMERIC_KEYS_INT = {"pack_count"}
NUMERIC_KEYS_FLOAT = {"capacity_liters", "capacity_oz", "weight_lb", "length_ft", "wattage_w"}
MULTI_KEYS = {"color", "size"}

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")
_PAREN = re.compile(r"\(([^()]*)\)")
_PACK = re.compile(
    r"(?<!\w)(?:pack of (\d{1,3})|set of (\d{1,3})|(\d{1,3})\s*-?\s*(?:pack|pk|pcs|pieces|count|ct)(?!\w)"
    # "3x" / "3 x" (not "2x4" or "3 x 4" dimensions, not "3XL" sizes) and "x3"
    r"|(\d{1,3})\s*x(?!\w)(?!\s*\d)|x(\d{1,3})(?!\w))",
    re.IGNORECASE,
)
_TAIL = re.compile(r"\s+[-–—|]\s+(?P<tail>[^-–—|]+?)\s*$")
_SIZE_TOKEN = re.compile(r"^(?:size\s*)?(\d{1,2}(?:\.5)?)$", re.IGNORECASE)
_CAPACITY = re.compile(
    r"(?<!\w)(\d+(?:\.\d+)?)\s*-?\s*(oz|ounce|ounces|l|liter|liters|litre|litres)(?!\w)", re.IGNORECASE
)
_PRICE_NUM = re.compile(r"\d[\d.,]*")
_THOUSANDS_ONLY = re.compile(r"\d{1,3}(?:([.,])\d{3})(?:\1\d{3})*")
_MATERIAL = re.compile(r"^(?:material|made of|made from|fabric|upper)\s*:?\s*(.+)$", re.IGNORECASE)
_CONNECTOR = re.compile(r"^connector\s*:?\s*(.+)$", re.IGNORECASE)
_KV = re.compile(r"^\s*([^=:]+?)\s*[=:]\s*(.*?)\s*$")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub(" ", text)
    return _WS.sub(" ", text).strip()


def build_abbreviation_regex(rules: Rules) -> re.Pattern[str] | None:
    if not rules.abbreviations:
        return None
    keys = sorted(rules.abbreviations, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(k) for k in keys) + r")(?!\w)", re.IGNORECASE)


def expand_abbreviations(text: str, rules: Rules, pattern: re.Pattern[str] | None) -> str:
    if pattern is None:
        return text
    return pattern.sub(lambda m: rules.abbreviations[m.group(0).lower()], text)


def normalize_size(token: str) -> str | None:
    t = token.strip().lower().replace("size", "").strip()
    if not t:
        return None
    if t in SIZE_ALIASES:
        return SIZE_ALIASES[t]
    up = t.upper()
    if up in APPAREL_SIZES:
        return up
    m = _SIZE_TOKEN.match(t)
    if m:
        num = m.group(1)
        return num[:-2] if num.endswith(".0") else num
    return None


def normalize_color(token: str) -> str | None:
    t = token.strip().lower()
    t = COLOR_ALIASES.get(t, t)
    return t if t in COLOR_WORDS else None


def extract_variations(title: str) -> tuple[str, list[str], list[str]]:
    """Pull `(Black/Navy, L)` style groups out of the title. Returns (title_without, colors, sizes)."""
    colors: list[str] = []
    sizes: list[str] = []

    def _consume(m: re.Match[str]) -> str:
        inner = m.group(1)
        recognised = False
        for seg in re.split(r"[,;]", inner):
            for tok in re.split(r"[/&+]| and ", seg):
                tok = tok.strip()
                if not tok:
                    continue
                c = normalize_color(tok)
                if c:
                    recognised = True
                    if c not in colors:
                        colors.append(c)
                    continue
                s = normalize_size(tok)
                if s:
                    recognised = True
                    if s not in sizes:
                        sizes.append(s)
        return " " if recognised else m.group(0)

    stripped = _PAREN.sub(_consume, title)
    return normalize_text(stripped), colors, sizes


def extract_pack_count(title: str) -> tuple[str, int | None]:
    m = _PACK.search(title)
    if not m:
        return title, None
    count = int(next(g for g in m.groups() if g))
    return normalize_text(title[: m.start()] + " " + title[m.end() :]), count


def extract_tail(title: str) -> tuple[str, list[str]]:
    m = _TAIL.search(title)
    if not m:
        return title, []
    tail = m.group("tail").strip()
    return normalize_text(title[: m.start()]), tail.split()


def extract_gender(text: str) -> str | None:
    for tok in re.findall(r"[a-z]+", text.lower()):
        if tok in GENDER_WORDS:
            return GENDER_WORDS[tok]
    return None


def extract_capacity(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for num, unit in _CAPACITY.findall(text):
        u = unit.lower()
        key = "capacity_oz" if u.startswith("o") else "capacity_liters"
        out.setdefault(key, float(num))
    return out


def split_bullets(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"\n|•|;|\s+-\s+|\s+\|\s+", raw)
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        p = normalize_text(p).strip("-•* ").strip()
        if not p:
            continue
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def split_tag_pairs(raw: str | None) -> list[tuple[str, str]]:
    """Split `k=v;k=v` / `k=v|k=v` / `k=v,k=v`. Comma is used only when it precedes another `key=`."""
    if not raw:
        return []
    raw = normalize_text(raw)
    if ";" in raw:
        chunks = raw.split(";")
    elif "|" in raw:
        chunks = raw.split("|")
    else:
        chunks = re.split(r",(?=\s*[A-Za-z_][\w\s-]*\s*[=:])", raw)
    pairs: list[tuple[str, str]] = []
    for ch in chunks:
        m = _KV.match(ch)
        if m and m.group(1) and m.group(2):
            pairs.append((m.group(1).strip().lower(), m.group(2).strip()))
    return pairs


def coerce_tag_value(key: str, value: str, supplier_id: str, rules: Rules) -> AttrValue | None:
    """Map a supplier tag value to a canonical attribute value; None if unusable."""
    if key in MULTI_KEYS:
        parts = [p.strip() for p in re.split(r"[/,&+]", value) if p.strip()]
        out: list[str] = []
        for p in parts:
            mapped = rules.tag_value(supplier_id, key, p)
            canon = normalize_size(mapped) if key == "size" else (normalize_color(mapped) or mapped.lower())
            if canon and canon not in out:
                out.append(canon)
        return out or None
    if key in NUMERIC_KEYS_INT:
        m = re.search(r"\d+", value)
        return int(m.group(0)) if m else None
    if key in NUMERIC_KEYS_FLOAT:
        m = re.search(r"\d+(?:\.\d+)?", value)
        return float(m.group(0)) if m else None
    return rules.tag_value(supplier_id, key, value)


def _price_token(token: str) -> str:
    """Normalise one numeric token to a plain decimal string.

    "1,299.99" / "1.299,99" -> "1299.99" (the last separator is the decimal point when both
    appear); "1,000" / "1.000.000" -> thousands groups; "12,50" -> "12.50"; "19.99" unchanged.
    A lone "1.299" stays a decimal (US style) - ambiguous, and dot-decimal is the common case.
    """
    t = token.rstrip(".,")
    if "." in t and "," in t:
        dec = "." if t.rfind(".") > t.rfind(",") else ","
        thousands = "," if dec == "." else "."
        return t.replace(thousands, "").replace(dec, ".")
    m = _THOUSANDS_ONLY.fullmatch(t)
    if m and (m.group(1) == "," or t.count(".") > 1):
        return t.replace(m.group(1), "")
    return t.replace(",", ".")


def parse_price(raw: str | None) -> tuple[Decimal | None, Decimal | None, bool]:
    """Returns (min, max, parse_ok). Ranges yield min<max; a single value yields min==max."""
    if raw is None or not raw.strip():
        return None, None, False
    nums = _PRICE_NUM.findall(raw)
    if not nums:
        return None, None, False
    values: list[Decimal] = []
    for n in nums[:2]:
        try:
            values.append(Decimal(_price_token(n)))
        except InvalidOperation:
            continue
    if not values:
        return None, None, False
    lo, hi = min(values), max(values)
    return lo, hi, True


def extract_material(bullets: list[str]) -> str | None:
    for b in bullets:
        m = _MATERIAL.match(b)
        if m:
            return normalize_text(m.group(1)).lower().rstrip(".")
    return None


def extract_connector(bullets: list[str]) -> str | None:
    for b in bullets:
        m = _CONNECTOR.match(b)
        if m:
            return normalize_text(m.group(1)).lower().rstrip(".")
    return None


def smart_title_case(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        text = text.lower()

    def cap(word: str) -> str:
        return "-".join(p[:1].upper() + p[1:] for p in word.split("-"))

    return " ".join(cap(w) for w in text.split(" "))
