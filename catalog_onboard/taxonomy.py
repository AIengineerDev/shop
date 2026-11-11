"""Canonical taxonomy loaded from canonical_categories. Fails fast on bad config."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property

from pydantic import TypeAdapter, ValidationError

from catalog_onboard.db.store import Store
from catalog_onboard.models import AttributeSpec

_SPECS = TypeAdapter(list[AttributeSpec])
_KW_SPLIT = re.compile(r"[,|;\n]")


class TaxonomyConfigError(ValueError):
    """Raised at load time when canonical_categories is malformed."""


@dataclass(frozen=True)
class CategoryNode:
    category_id: str
    name: str
    parent_id: str | None
    required_attributes: tuple[AttributeSpec, ...]
    prohibited_keywords: tuple[str, ...]


def parse_prohibited(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    seen: dict[str, None] = {}
    for part in _KW_SPLIT.split(raw):
        kw = part.strip().lower()
        if kw:
            seen.setdefault(kw, None)
    return tuple(seen)


def parse_attribute_specs(category_id: str, raw: str | None) -> tuple[AttributeSpec, ...]:
    if raw is None or raw.strip() == "":
        return ()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TaxonomyConfigError(f"{category_id}: required_attributes_json is not valid JSON: {exc}") from exc
    try:
        specs = _SPECS.validate_python(data)
    except ValidationError as exc:
        raise TaxonomyConfigError(f"{category_id}: invalid attribute spec: {exc}") from exc
    keys = [s.key for s in specs]
    if len(keys) != len(set(keys)):
        raise TaxonomyConfigError(f"{category_id}: duplicate attribute keys {keys}")
    return tuple(specs)


@dataclass
class Taxonomy:
    nodes: dict[str, CategoryNode]
    children: dict[str | None, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for node in self.nodes.values():
            if node.parent_id is not None and node.parent_id not in self.nodes:
                raise TaxonomyConfigError(f"{node.category_id}: parent {node.parent_id!r} does not exist")
            self.children.setdefault(node.parent_id, []).append(node.category_id)
        for cid in self.nodes:
            self.ancestors(cid)  # cycle detection

    @classmethod
    def from_store(cls, store: Store) -> Taxonomy:
        nodes: dict[str, CategoryNode] = {}
        for row in store.iter_categories():
            cid = str(row["category_id"])
            if cid in nodes:
                raise TaxonomyConfigError(f"duplicate category_id {cid}")
            nodes[cid] = CategoryNode(
                category_id=cid,
                name=str(row["name"]),
                parent_id=None if row["parent_id"] in (None, "") else str(row["parent_id"]),
                required_attributes=parse_attribute_specs(cid, row["required_attributes_json"]),
                prohibited_keywords=parse_prohibited(row["prohibited_keywords"]),
            )
        if not nodes:
            raise TaxonomyConfigError("canonical_categories is empty")
        return cls(nodes=nodes)

    def __contains__(self, category_id: object) -> bool:
        return category_id in self.nodes

    def get(self, category_id: str) -> CategoryNode:
        try:
            return self.nodes[category_id]
        except KeyError as exc:
            raise KeyError(f"unknown category {category_id!r}") from exc

    def is_leaf(self, category_id: str) -> bool:
        return category_id in self.nodes and not self.children.get(category_id)

    @cached_property
    def leaf_ids(self) -> tuple[str, ...]:
        return tuple(sorted(cid for cid in self.nodes if self.is_leaf(cid)))

    def ancestors(self, category_id: str) -> list[CategoryNode]:
        """Root-first list of ancestors, excluding the node itself."""
        out: list[CategoryNode] = []
        seen: set[str] = set()
        cur = self.get(category_id).parent_id
        while cur is not None:
            if cur in seen:
                raise TaxonomyConfigError(f"cycle detected at {cur}")
            seen.add(cur)
            node = self.get(cur)
            out.append(node)
            cur = node.parent_id
        out.reverse()
        return out

    def path(self, category_id: str) -> str:
        names = [n.name for n in self.ancestors(category_id)] + [self.get(category_id).name]
        return " > ".join(names)

    def effective_prohibited(self, category_id: str) -> tuple[str, ...]:
        """Category keywords plus all ancestors', deduplicated, order root-first."""
        seen: dict[str, None] = {}
        for node in [*self.ancestors(category_id), self.get(category_id)]:
            for kw in node.prohibited_keywords:
                seen.setdefault(kw, None)
        return tuple(seen)

    def required_attributes(self, category_id: str) -> tuple[AttributeSpec, ...]:
        return self.get(category_id).required_attributes
