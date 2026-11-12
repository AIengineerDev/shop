"""Stage 6: the single routing rule."""

from __future__ import annotations

from catalog_onboard.models import Status, ValidationResult


def route(result: ValidationResult) -> Status:
    return "draft" if result.ok else "quarantined"
