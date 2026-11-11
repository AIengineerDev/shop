from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Query, Response

from repricer.api.app import ConnDep, SettingsDep, StoreDep, build_extractors
from repricer.api.schemas import (
    ActionOut,
    BreakOut,
    ChangeOut,
    OfferOut,
    OverrideRequest,
    Page,
    PoliciesOut,
    RecommendationOut,
    ReviewCard,
    ReviewRequest,
    ReviewResult,
    RunOut,
    RunRequest,
    VerifyOut,
    money,
)
from repricer.clock import utcnow
from repricer.db import repo
from repricer.execution import audit, review, runner
from repricer.extraction.pipeline import extract_stage
from repricer.models import ReviewStatus
from repricer.money import fmt, to_cents

router = APIRouter()


# --- runs -------------------------------------------------------------------


@router.post("/runs", status_code=202, response_model=RunOut)
def start_run(body: RunRequest, conn: ConnDep, settings: SettingsDep, store: StoreDep) -> RunOut:
    primary, fallback = build_extractors(settings)
    now = utcnow()

    def extract(c: Any, _run_id: str) -> int:
        summary = extract_stage(
            c,
            extractor=primary,
            now=now,
            batch_size=settings.llm_batch,
            fallback=fallback,
        )
        return summary.scrapes

    result = runner.run(
        conn,
        store=store,
        trigger="api",
        now=now,
        hard_cap_bp=settings.hard_cap_bp,
        extract=extract,
        stages=tuple(body.stages),
    )
    return RunOut.from_row(repo.get_run(conn, result.run_id))


@router.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str, conn: ConnDep) -> RunOut:
    return RunOut.from_row(repo.get_run(conn, run_id))


# --- recommendations --------------------------------------------------------


@router.get("/recommendations", response_model=Page)
def list_recommendations(
    conn: ConnDep,
    status: str | None = None,
    sku: str | None = None,
    run_id: str | None = None,
    decision: str | None = None,
    min_abs_delta_bp: int | None = Query(default=None, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Page:
    rows, total = repo.list_recommendations(
        conn,
        status=status,
        sku=sku,
        run_id=run_id,
        decision=decision,
        min_abs_delta_bp=min_abs_delta_bp,
        limit=limit,
        offset=offset,
    )
    return Page(
        items=[RecommendationOut.from_row(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/recommendations/{rec_id}", response_model=ReviewCard)
def review_card(rec_id: str, conn: ConnDep) -> ReviewCard:
    r = repo.get_recommendation(conn, rec_id)
    inv = repo.get_inventory(conn, r["sku"])
    base = RecommendationOut.from_row(r)
    offers = repo.offers_by_ids(conn, json.loads(r["competitor_offer_ids"]))
    actions = [
        ActionOut(
            action_id=a["action_id"],
            action=a["action"],
            actor=a["actor"],
            override_price=money(a["override_price_cents"]),
            note=a["note"],
            created_at=a["created_at"],
        )
        for a in repo.actions_for_recommendation(conn, rec_id)
    ]
    return ReviewCard(
        **base.model_dump(),
        title=inv.title,
        stock_count=inv.stock_count,
        price_class=inv.price_class,
        eligible_offers=[OfferOut.from_row(o) for o in offers],
        actions=actions,
    )


def _result(conn: Any, rec_id: str, change_id: str | None) -> ReviewResult:
    r = repo.get_recommendation(conn, rec_id)
    inv = repo.get_inventory(conn, r["sku"])
    return ReviewResult(
        recommendation_id=rec_id,
        review_status=r["review_status"],
        change_id=change_id,
        sku=r["sku"],
        new_price=fmt(inv.current_price_cents)
        if r["review_status"] == ReviewStatus.APPLIED.value
        else None,
    )


@router.post("/recommendations/{rec_id}/approve", response_model=ReviewResult)
def approve(rec_id: str, body: ReviewRequest, conn: ConnDep, settings: SettingsDep) -> ReviewResult:
    change_id = review.approve(
        conn,
        rec_id,
        actor=body.actor,
        note=body.note,
        now=utcnow(),
        hard_cap_bp=settings.hard_cap_bp,
    )
    return _result(conn, rec_id, change_id)


@router.post("/recommendations/{rec_id}/reject", response_model=ReviewResult)
def reject(rec_id: str, body: ReviewRequest, conn: ConnDep) -> ReviewResult:
    review.reject(conn, rec_id, actor=body.actor, note=body.note, now=utcnow())
    return _result(conn, rec_id, None)


@router.post("/recommendations/{rec_id}/override", response_model=ReviewResult)
def override(
    rec_id: str, body: OverrideRequest, conn: ConnDep, settings: SettingsDep
) -> ReviewResult:
    change_id = review.override(
        conn,
        rec_id,
        actor=body.actor,
        price_cents=to_cents(body.price),
        note=body.note,
        now=utcnow(),
        hard_cap_bp=settings.hard_cap_bp,
    )
    return _result(conn, rec_id, change_id)


# --- skus -------------------------------------------------------------------


@router.get("/skus/{sku}/history", response_model=list[ChangeOut])
def sku_history(sku: str, conn: ConnDep) -> list[ChangeOut]:
    repo.get_inventory(conn, sku)  # 404 if unknown
    return [ChangeOut.from_dict(d) for d in audit.history(conn, sku)]


@router.get("/skus/{sku}/offers", response_model=list[OfferOut])
def sku_offers(sku: str, conn: ConnDep) -> list[OfferOut]:
    repo.get_inventory(conn, sku)
    return [OfferOut.from_row(o) for o in repo.offers_for_sku(conn, sku)]


# --- audit / policies -------------------------------------------------------


@router.get("/audit/verify", response_model=VerifyOut)
def verify(conn: ConnDep, response: Response) -> VerifyOut:
    breaks = audit.verify_chain(conn)
    if breaks:
        response.status_code = 409
    return VerifyOut(ok=not breaks, breaks=[BreakOut(**b.__dict__) for b in breaks])


@router.get("/policies", response_model=PoliciesOut)
def policies(store: StoreDep) -> PoliciesOut:
    return PoliciesOut(
        version=store.version,
        policies={k: v.model_dump(mode="json") for k, v in store.policies.items()},
    )


@router.post("/policies/reload", response_model=PoliciesOut)
def reload_policies(store: StoreDep) -> PoliciesOut:
    store.reload()  # PolicyError -> 422, old version stays
    return policies(store)
