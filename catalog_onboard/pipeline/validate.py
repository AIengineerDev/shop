"""Stage 5a: attribute validation against the category spec. Pure Python, table-driven."""

from __future__ import annotations

import re
from collections.abc import Sequence

from catalog_onboard.models import AttributeSpec, AttrValue, ReasonCode


def _blocking(code: str, detail: str | None = None) -> ReasonCode:
    return ReasonCode(code=code, severity="blocking", stage="validate", detail=detail)


def _match_enum(value: object, allowed: Sequence[str]) -> str | None:
    s = str(value).strip()
    for a in allowed:
        if a.lower() == s.lower():
            return a
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        m = re.fullmatch(r"\s*(-?\d+)\s*", value)
        if m:
            return int(m.group(1))
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0"):
            return False
    return None


def validate_attributes(
    specs: Sequence[AttributeSpec], attrs: dict[str, AttrValue]
) -> tuple[dict[str, AttrValue], list[ReasonCode]]:
    """Returns (coerced attributes, reason codes). Any blocking code means the listing is not publishable."""
    codes: list[ReasonCode] = []
    out: dict[str, AttrValue] = {}
    for spec in specs:
        key = spec.key
        if key not in attrs or attrs[key] is None or attrs[key] == [] or attrs[key] == "":
            codes.append(_blocking(f"ATTR_MISSING:{key}"))
            continue
        value: object = attrs[key]

        # list-ness vs multi
        if isinstance(value, list):
            if not spec.multi:
                if len(value) == 1:
                    value = value[0]
                else:
                    codes.append(_blocking(f"ATTR_MULTI_NOT_ALLOWED:{key}", f"got {value!r}"))
                    continue
        elif spec.multi:
            value = [value]

        if spec.type == "enum":
            allowed = spec.allowed or []
            if spec.multi:
                assert isinstance(value, list)
                matched: list[str] = []
                bad: list[str] = []
                for v in value:
                    m = _match_enum(v, allowed)
                    (matched if m else bad).append(m or str(v))
                if bad:
                    codes.append(_blocking(f"ATTR_ENUM_INVALID:{key}", f"{bad!r} not in allowed"))
                    continue
                out[key] = matched
            else:
                m = _match_enum(value, allowed)
                if m is None:
                    codes.append(_blocking(f"ATTR_ENUM_INVALID:{key}", f"{value!r} not in allowed"))
                    continue
                out[key] = m
        elif spec.type in ("int", "float"):
            num = _coerce_int(value) if spec.type == "int" else _coerce_float(value)
            if num is None:
                codes.append(_blocking(f"ATTR_RANGE:{key}", f"{value!r} is not a number"))
                continue
            if (spec.min is not None and num < spec.min) or (spec.max is not None and num > spec.max):
                codes.append(_blocking(f"ATTR_RANGE:{key}", f"{num} outside [{spec.min}, {spec.max}]"))
                continue
            out[key] = num
        elif spec.type == "bool":
            b = _coerce_bool(value)
            if b is None:
                codes.append(_blocking(f"ATTR_ENUM_INVALID:{key}", f"{value!r} is not boolean"))
                continue
            out[key] = b
        else:  # string
            s = str(value).strip()
            if spec.max_len is not None and len(s) > spec.max_len:
                codes.append(_blocking(f"ATTR_TOO_LONG:{key}", f"{len(s)} > {spec.max_len}"))
                continue
            if spec.pattern is not None and not re.fullmatch(spec.pattern, s):
                codes.append(_blocking(f"ATTR_PATTERN:{key}", f"{s!r} !~ {spec.pattern}"))
                continue
            out[key] = s
    return out, codes
