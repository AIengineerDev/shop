"""Policy YAML loading with atomic swap on reload."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from repricer.models import Policy

DEFAULT_CLASS = "default"


class PolicyError(ValueError):
    pass


def load_policies(path: Path) -> dict[str, Policy]:
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as e:
        raise PolicyError(f"cannot read policies at {path}: {e}") from e
    if not isinstance(raw, dict) or not raw:
        raise PolicyError("policies file must be a non-empty mapping of price_class -> policy")
    out: dict[str, Policy] = {}
    for price_class, body in raw.items():
        if not isinstance(price_class, str) or not price_class:
            raise PolicyError(f"invalid price_class key {price_class!r}")
        if not isinstance(body, dict):
            raise PolicyError(f"policy {price_class!r} must be a mapping")
        try:
            out[price_class] = Policy.model_validate(body)
        except ValidationError as e:
            raise PolicyError(f"policy {price_class!r}: {e}") from e
    if DEFAULT_CLASS not in out:
        raise PolicyError(f"policies must define a {DEFAULT_CLASS!r} class")
    return out


class PolicyStore:
    """Holds the active policy set. ``reload`` only swaps on success (AC-A4)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._policies = load_policies(path)
        self._version = 1

    @property
    def version(self) -> int:
        return self._version

    @property
    def policies(self) -> dict[str, Policy]:
        return dict(self._policies)

    def for_class(self, price_class: str) -> Policy:
        return self._policies.get(price_class, self._policies[DEFAULT_CLASS])

    def reload(self) -> int:
        fresh = load_policies(self.path)  # raises PolicyError; current set untouched
        self._policies = fresh
        self._version += 1
        return self._version
