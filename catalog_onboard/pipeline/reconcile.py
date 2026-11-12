"""Stage 4: merge deterministic attributes with LLM proposals. Deterministic wins."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from catalog_onboard.models import AttributeSpec, AttrValue, ReasonCode

# Deterministic defaults for attributes a listing omits because the default is implied.
DEFAULTS: dict[str, AttrValue] = {"pack_count": 1}


def apply_defaults(specs: Sequence[AttributeSpec], attrs: dict[str, AttrValue]) -> list[ReasonCode]:
    codes: list[ReasonCode] = []
    for spec in specs:
        if spec.key not in attrs and spec.key in DEFAULTS:
            attrs[spec.key] = DEFAULTS[spec.key]
            codes.append(
                ReasonCode(
                    code=f"ATTR_DEFAULTED:{spec.key}",
                    severity="info",
                    stage="reconcile",
                    detail=str(DEFAULTS[spec.key]),
                )
            )
    return codes


def missing_specs(specs: Sequence[AttributeSpec], det_attrs: dict[str, AttrValue]) -> list[AttributeSpec]:
    """Specs the LLM should be asked about: not extracted deterministically and without a default."""
    return [s for s in specs if s.key not in det_attrs and s.key not in DEFAULTS]


def merge_attributes(
    specs: Sequence[AttributeSpec],
    det_attrs: dict[str, AttrValue],
    llm_attrs: dict[str, Any] | None,
) -> tuple[dict[str, AttrValue], dict[str, AttrValue], list[ReasonCode]]:
    """Returns (attributes for required keys, extra attributes, reason codes)."""
    codes: list[ReasonCode] = []
    required = {s.key for s in specs}
    attrs: dict[str, AttrValue] = {}
    for key, val in det_attrs.items():
        if key in required:
            attrs[key] = val
    if llm_attrs:
        for key, val in llm_attrs.items():
            if val is None or key not in required:
                continue
            if key in attrs:
                if attrs[key] != val:
                    codes.append(
                        ReasonCode(
                            code=f"ATTR_CONFLICT:{key}",
                            severity="warning",
                            stage="reconcile",
                            detail=f"deterministic {attrs[key]!r} kept over llm {val!r}",
                        )
                    )
                continue
            attrs[key] = val
    codes.extend(apply_defaults(specs, attrs))
    extra = {k: v for k, v in det_attrs.items() if k not in required}
    for k in extra:
        codes.append(ReasonCode(code=f"ATTR_EXTRA:{k}", severity="info", stage="reconcile"))
    return attrs, extra, codes
