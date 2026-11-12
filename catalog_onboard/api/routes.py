"""Read-only inspection endpoints.

No `from __future__ import annotations` here: FastAPI must evaluate the function-local Annotated aliases eagerly.
"""

from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel

from catalog_onboard.api.pagination import clamp_limit, decode_cursor, encode_cursor
from catalog_onboard.db.store import Store
from catalog_onboard.models import AttributeSpec, DraftListing, PipelineRunRecord, QuarantineItem
from catalog_onboard.taxonomy import Taxonomy


# ---------------------------------------------------------------- response models
class Health(BaseModel):
    status: str
    rules_version: str
    db_path: str


class DraftPage(BaseModel):
    items: list[DraftListing]
    next_cursor: str | None


class DraftDetail(BaseModel):
    listing: DraftListing
    trace: list[PipelineRunRecord]


class QuarantinePage(BaseModel):
    items: list[QuarantineItem]
    next_cursor: str | None


class QuarantineDetail(BaseModel):
    item: QuarantineItem
    trace: list[PipelineRunRecord]


class Trace(BaseModel):
    feed_item_id: str
    runs: list[PipelineRunRecord]


class TaxonomyNode(BaseModel):
    category_id: str
    name: str
    parent_id: str | None
    is_leaf: bool
    path: str
    required_attributes: list[AttributeSpec]
    prohibited_keywords: list[str]
    children: list[str]


class TaxonomyNodeDetail(TaxonomyNode):
    effective_prohibited_keywords: list[str]


class TaxonomyTree(BaseModel):
    nodes: list[TaxonomyNode]
    leaf_count: int


class ReasonCodeCount(BaseModel):
    code: str
    count: int


class SupplierStats(BaseModel):
    supplier_id: str
    draft: int
    quarantined: int
    quarantine_rate: float


class Stats(BaseModel):
    draft: int
    quarantined: int
    top_reason_codes: list[ReasonCodeCount]
    per_supplier: list[SupplierStats]


class ErrorBody(BaseModel):
    detail: str
    code: str



def as_json(model: BaseModel) -> Response:
    """Serialise once, with pydantic-core, on the worker thread.

    Returning a Response makes FastAPI skip re-validating the response model (it still documents
    it in OpenAPI via response_model=). Under concurrent load the default path re-validated and
    JSON-encoded large pages on the event loop: 8 clients got 4x less throughput than one."""
    return Response(content=model.model_dump_json(), media_type="application/json")

def not_found(what: str, ident: str | int) -> HTTPException:
    return HTTPException(status_code=404, detail={"detail": f"{what} {ident!r} not found", "code": "NOT_FOUND"})


# ------------------------------------------------------------------------ routes
def build_router(
    get_store: Callable[[], Store], get_taxonomy: Callable[[], Taxonomy], rules_version: str, db_path: str
) -> APIRouter:
    router = APIRouter()
    StoreDep = Annotated[Store, Depends(get_store)]
    TaxDep = Annotated[Taxonomy, Depends(get_taxonomy)]
    LimitQ = Annotated[int | None, Query(ge=1, le=500)]
    err: dict[int | str, dict[str, Any]] = {404: {"model": ErrorBody}}

    @router.get("/health", response_model=Health)
    def health() -> Response:
        return as_json(Health(status="ok", rules_version=rules_version, db_path=db_path))

    @router.get("/catalog/draft", response_model=DraftPage)
    def list_draft(
        store: StoreDep,
        supplier_id: str | None = None,
        category_id: str | None = None,
        q: str | None = None,
        limit: LimitQ = None,
        cursor: str | None = None,
    ) -> Response:
        n = clamp_limit(limit)
        rows = store.list_drafts(
            supplier_id=supplier_id, category_id=category_id, q=q, after=decode_cursor(cursor), limit=n + 1
        )
        nxt = None
        if len(rows) > n:
            rows = rows[:n]
            last = rows[-1]
            nxt = encode_cursor(last.created_at.isoformat(timespec="milliseconds"), last.listing_id)
        return as_json(DraftPage(items=rows, next_cursor=nxt))

    @router.get("/catalog/draft/{listing_id}", response_model=DraftDetail, responses=err)
    def get_draft(listing_id: int, store: StoreDep) -> Response:
        d = store.get_draft(listing_id)
        if d is None:
            raise not_found("draft listing", listing_id)
        return as_json(DraftDetail(listing=d, trace=store.trace(d.feed_item_id)))

    @router.get("/quarantine", response_model=QuarantinePage)
    def list_quarantine(
        store: StoreDep,
        supplier_id: str | None = None,
        reason_code: str | None = None,
        proposed_category_id: str | None = None,
        limit: LimitQ = None,
        cursor: str | None = None,
    ) -> Response:
        n = clamp_limit(limit)
        rows = store.list_quarantine(
            supplier_id=supplier_id,
            reason_code=reason_code,
            proposed_category_id=proposed_category_id,
            after=decode_cursor(cursor),
            limit=n + 1,
        )
        nxt = None
        if len(rows) > n:
            rows = rows[:n]
            last = rows[-1]
            nxt = encode_cursor(last.created_at.isoformat(timespec="milliseconds"), last.quarantine_id)
        return as_json(QuarantinePage(items=rows, next_cursor=nxt))

    @router.get("/quarantine/{quarantine_id}", response_model=QuarantineDetail, responses=err)
    def get_quarantine(quarantine_id: int, store: StoreDep) -> Response:
        item = store.get_quarantine(quarantine_id)
        if item is None:
            raise not_found("quarantine item", quarantine_id)
        return as_json(QuarantineDetail(item=item, trace=store.trace(item.feed_item_id)))

    @router.get("/items/{feed_item_id}/trace", response_model=Trace, responses=err)
    def get_trace(feed_item_id: str, store: StoreDep) -> Response:
        runs = store.trace(feed_item_id)
        if not runs:
            raise not_found("feed item", feed_item_id)
        return as_json(Trace(feed_item_id=feed_item_id, runs=runs))

    def _node(tax: Taxonomy, cid: str) -> TaxonomyNode:
        node = tax.get(cid)
        return TaxonomyNode(
            category_id=cid,
            name=node.name,
            parent_id=node.parent_id,
            is_leaf=tax.is_leaf(cid),
            path=tax.path(cid),
            required_attributes=list(node.required_attributes),
            prohibited_keywords=list(node.prohibited_keywords),
            children=sorted(tax.children.get(cid, [])),
        )

    @router.get("/taxonomy", response_model=TaxonomyTree)
    def taxonomy_tree(tax: TaxDep) -> Response:
        nodes = [_node(tax, cid) for cid in sorted(tax.nodes)]
        return as_json(TaxonomyTree(nodes=nodes, leaf_count=len(tax.leaf_ids)))

    @router.get("/taxonomy/{category_id}", response_model=TaxonomyNodeDetail, responses=err)
    def taxonomy_node(category_id: str, tax: TaxDep) -> Response:
        if category_id not in tax:
            raise not_found("category", category_id)
        base = _node(tax, category_id)
        return as_json(TaxonomyNodeDetail(
            **base.model_dump(), effective_prohibited_keywords=list(tax.effective_prohibited(category_id))
        ))

    @router.get("/stats", response_model=Stats)
    def stats(store: StoreDep) -> Response:
        return as_json(Stats(**store.stats()))

    return router
