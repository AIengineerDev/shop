"""Deterministic rule tables, loaded from YAML and hashed into a rules_version."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

RULE_FILES = ("abbreviations.yaml", "supplier_tag_map.yaml", "global_prohibited.yaml")


@dataclass(frozen=True)
class Rules:
    abbreviations: dict[str, str]
    """lowercase token/phrase -> replacement"""
    supplier_tag_keys: dict[str, dict[str, str]]
    """supplier_id -> {supplier key -> canonical key}; '*' applies to all suppliers"""
    supplier_tag_values: dict[str, dict[str, dict[str, str]]]
    """supplier_id -> {canonical key -> {supplier value -> canonical value}}"""
    global_prohibited: tuple[str, ...]
    version: str

    def tag_key(self, supplier_id: str, key: str) -> str | None:
        k = key.strip().lower()
        for scope in (supplier_id, "*"):
            mapped = self.supplier_tag_keys.get(scope, {}).get(k)
            if mapped:
                return mapped
        return None

    def tag_value(self, supplier_id: str, canonical_key: str, value: str) -> str:
        v = value.strip().lower()
        for scope in (supplier_id, "*"):
            mapped = self.supplier_tag_values.get(scope, {}).get(canonical_key, {}).get(v)
            if mapped:
                return mapped
        return v


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def compute_rules_version(rules_dir: Path) -> str:
    h = hashlib.sha256()
    for name in RULE_FILES:
        h.update(name.encode())
        h.update((rules_dir / name).read_bytes())
    return h.hexdigest()[:16]


def load_rules(rules_dir: Path) -> Rules:
    missing = [n for n in RULE_FILES if not (rules_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"rules dir {rules_dir} missing: {missing}")

    abbreviations_raw = _load_yaml(rules_dir / "abbreviations.yaml")
    if not isinstance(abbreviations_raw, dict):
        raise ValueError("abbreviations.yaml must be a mapping")
    abbreviations = {str(k).lower(): str(v) for k, v in abbreviations_raw.items()}

    tag_map_raw = _load_yaml(rules_dir / "supplier_tag_map.yaml")
    if not isinstance(tag_map_raw, dict):
        raise ValueError("supplier_tag_map.yaml must be a mapping")
    keys: dict[str, dict[str, str]] = {}
    values: dict[str, dict[str, dict[str, str]]] = {}
    for supplier, spec in tag_map_raw.items():
        spec = spec or {}
        if not isinstance(spec, dict):
            raise ValueError(f"supplier_tag_map[{supplier!r}] must be a mapping")
        keys[str(supplier)] = {str(k).lower(): str(v) for k, v in (spec.get("keys") or {}).items()}
        values[str(supplier)] = {
            str(ck): {str(sv).lower(): str(cv) for sv, cv in (vm or {}).items()}
            for ck, vm in (spec.get("values") or {}).items()
        }

    global_raw = _load_yaml(rules_dir / "global_prohibited.yaml")
    if isinstance(global_raw, dict):
        global_raw = global_raw.get("keywords", [])
    if not isinstance(global_raw, list):
        raise ValueError("global_prohibited.yaml must be a list or {keywords: [...]}")
    global_prohibited = tuple(str(k).strip().lower() for k in global_raw if str(k).strip())

    return Rules(
        abbreviations=abbreviations,
        supplier_tag_keys=keys,
        supplier_tag_values=values,
        global_prohibited=global_prohibited,
        version=compute_rules_version(rules_dir),
    )
