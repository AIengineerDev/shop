"""Call -> parse -> validate -> retry loop. Returns typed results or None plus the full trace."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from catalog_onboard.models import (
    AttributeSpec,
    Candidate,
    LLMAttempt,
    LLMClassification,
    NormalizedListing,
)
from catalog_onboard.pipeline.llm import LLMClient, LLMRequest, LLMTransportError, Purpose, prompts
from catalog_onboard.pipeline.llm.schema import (
    build_attribute_model,
    build_classification_model,
    json_schema_for,
)

M = TypeVar("M", bound=BaseModel)


def run_structured(
    client: LLMClient,
    *,
    purpose: Purpose,
    system: str,
    user: str,
    model: type[M],
    hints: dict[str, object],
    max_retries: int,
) -> tuple[M | None, list[LLMAttempt]]:
    attempts: list[LLMAttempt] = []
    schema = json_schema_for(model)
    prompt = user
    for i in range(max_retries + 1):
        req = LLMRequest(purpose=purpose, system=system, user=prompt, json_schema=schema, hints=hints)
        raw: str | None = None
        try:
            raw = client.complete(req)
            data = json.loads(raw)
            parsed = model.model_validate(data)
        except LLMTransportError as exc:
            attempts.append(
                LLMAttempt(purpose=purpose, attempt=i, raw_response=None, error=f"transport: {exc}", ok=False)
            )
            continue
        except json.JSONDecodeError as exc:
            err = f"invalid JSON: {exc}"
            attempts.append(LLMAttempt(purpose=purpose, attempt=i, raw_response=raw, error=err, ok=False))
            prompt = user + prompts.retry_suffix(err)
            continue
        except ValidationError as exc:
            err = "schema violation: " + "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            attempts.append(LLMAttempt(purpose=purpose, attempt=i, raw_response=raw, error=err, ok=False))
            prompt = user + prompts.retry_suffix(err)
            continue
        attempts.append(LLMAttempt(purpose=purpose, attempt=i, raw_response=raw, error=None, ok=True))
        return parsed, attempts
    return None, attempts


def classify(
    client: LLMClient,
    listing: NormalizedListing,
    candidates: Sequence[Candidate],
    *,
    max_retries: int,
) -> tuple[LLMClassification | None, list[LLMAttempt]]:
    ids = [c.category_id for c in candidates]
    model = build_classification_model(ids)
    return run_structured(
        client,
        purpose="classify",
        system=prompts.CLASSIFY_SYSTEM,
        user=prompts.classify_user(listing, candidates),
        model=model,
        hints={"candidate_ids": ids},
        max_retries=max_retries,
    )


def extract(
    client: LLMClient,
    listing: NormalizedListing,
    specs: Sequence[AttributeSpec],
    *,
    max_retries: int,
) -> tuple[BaseModel | None, list[LLMAttempt]]:
    model = build_attribute_model(specs)
    return run_structured(
        client,
        purpose="extract",
        system=prompts.EXTRACT_SYSTEM,
        user=prompts.extract_user(listing, specs),
        model=model,
        hints={"text": listing.search_text, "specs": [s.model_dump() for s in specs]},
        max_retries=max_retries,
    )
