"""Dynamic Pydantic models derived from the taxonomy, so out-of-vocabulary answers fail to parse."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, create_model

from catalog_onboard.models import AttributeSpec, LLMClassification


def build_classification_model(candidate_ids: Sequence[str]) -> type[LLMClassification]:
    if not candidate_ids:
        raise ValueError("need at least one candidate")
    lit: Any = Literal[tuple(candidate_ids)]
    model: type[LLMClassification] = create_model(
        "ConstrainedClassification",
        __base__=LLMClassification,
        category_id=(lit, ...),
    )
    return model


def _field_type(spec: AttributeSpec) -> Any:
    if spec.type == "enum":
        lit = Literal[tuple(spec.allowed or ())]  # type: ignore[valid-type]
        return list[lit] if spec.multi else lit  # type: ignore[valid-type]
    if spec.type == "int":
        kw: dict[str, Any] = {}
        if spec.min is not None:
            kw["ge"] = int(spec.min)
        if spec.max is not None:
            kw["le"] = int(spec.max)
        return Annotated[int, Field(**kw)]
    if spec.type == "float":
        kw = {}
        if spec.min is not None:
            kw["ge"] = spec.min
        if spec.max is not None:
            kw["le"] = spec.max
        return Annotated[float, Field(**kw)]
    if spec.type == "bool":
        return bool
    kw2: dict[str, Any] = {}
    if spec.max_len is not None:
        kw2["max_length"] = spec.max_len
    if spec.pattern is not None:
        kw2["pattern"] = spec.pattern
    return Annotated[str, StringConstraints(**kw2)]


class _AttrBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


def build_attribute_model(specs: Sequence[AttributeSpec]) -> type[BaseModel]:
    """Every field is optional: the model returns null when the listing does not state the attribute."""
    fields: dict[str, Any] = {}
    for spec in specs:
        fields[spec.key] = (_field_type(spec) | None, Field(default=None))
    return create_model("ExtractedAttributes", __base__=_AttrBase, **fields)


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    # Structured outputs require every property to be listed in `required`; optional fields are nullable.
    schema["required"] = list(schema.get("properties", {}).keys())
    return schema
