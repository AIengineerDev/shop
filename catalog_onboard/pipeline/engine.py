"""Orchestrates the stages per item, persists the trail, and enforces idempotency."""

from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from catalog_onboard import obs
from catalog_onboard.config import Settings
from catalog_onboard.db.store import Store
from catalog_onboard.models import (
    AttrValue,
    Candidate,
    CandidateListing,
    LLMAttempt,
    LLMClassification,
    NormalizedListing,
    RawFeedRow,
    ReasonCode,
    Status,
    ValidationResult,
)
from catalog_onboard.pipeline import candidates as cand_mod
from catalog_onboard.pipeline import clean, comply, reconcile, route, validate
from catalog_onboard.pipeline.llm import LLMClient, runner
from catalog_onboard.rules import Rules, pipeline_version
from catalog_onboard.taxonomy import Taxonomy

log = logging.getLogger("catalog_onboard.engine")


@dataclass
class ItemOutcome:
    feed_item_id: str
    status: Status | None
    skipped: bool = False
    error: str | None = None
    reason_codes: list[ReasonCode] = field(default_factory=list)


@dataclass
class BatchSummary:
    processed: int = 0
    draft: int = 0
    quarantined: int = 0
    errors: int = 0
    skipped: int = 0
    run_id: str = ""


class _Trail:
    """Collects per-stage records and flushes them to pipeline_runs."""

    def __init__(self, store: Store, run_id: str, feed_item_id: str):
        self.store = store
        self.run_id = run_id
        self.feed_item_id = feed_item_id
        self.seq = 0

    def record(self, stage: str, input_obj: Any, output_obj: Any, started: float, llm_raw: Any = None) -> None:
        if isinstance(input_obj, list):
            input_obj = {"items": input_obj}
        if isinstance(output_obj, list):
            output_obj = {"items": output_obj}
        self.seq += 1
        self.store.insert_run(
            run_id=self.run_id,
            feed_item_id=self.feed_item_id,
            stage=stage,
            seq=self.seq,
            input_obj=_dump(input_obj),
            output_obj=_dump(output_obj),
            llm_raw=_dump(llm_raw),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


def _dump(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, list):
        return [_dump(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items()}
    return obj


def llm_failure_code(trace: list[LLMAttempt], *, candidate_ids: set[str] | None = None) -> str:
    """Reason code for an LLM call that never produced a valid result (AC-2.2, FR-2a).

    - every attempt failed in transport (timeout, 5xx, refusal) -> LLM_TRANSPORT_ERROR
    - every parseable attempt named a category outside the shortlist -> CLASSIFICATION_OUT_OF_SET
    - anything else (invalid JSON, schema violations, mixed) -> LLM_SCHEMA_INVALID
    """
    failed = [a for a in trace if not a.ok]
    if failed and all((a.error or "").startswith("transport:") for a in failed):
        return "LLM_TRANSPORT_ERROR"
    if candidate_ids is not None and failed:

        def out_of_set(a: LLMAttempt) -> bool:
            try:
                data = json.loads(a.raw_response or "")
            except ValueError:
                return False
            return isinstance(data, dict) and data.get("category_id") not in candidate_ids

        if all(out_of_set(a) for a in failed):
            return "CLASSIFICATION_OUT_OF_SET"
    return "LLM_SCHEMA_INVALID"


UPSTREAM_STAGES = frozenset({"clean", "candidates", "classify"})


def stored_llm_attrs(trace: list[LLMAttempt]) -> dict[str, Any] | None:
    """The attributes the LLM extracted at ingest time (last successful extract attempt)."""
    for attempt in reversed(trace):
        if attempt.purpose == "extract" and attempt.ok and attempt.raw_response:
            data = json.loads(attempt.raw_response)
            return data if isinstance(data, dict) else None
    return None


def _log_outcome(outcome: ItemOutcome) -> None:
    if outcome.skipped:
        log.debug("item skipped (already processed)", extra={"event": "item_skipped"})
        return
    log.info(
        "item %s", outcome.status,
        extra={"event": "item_done", "status": outcome.status, "codes": [c.code for c in outcome.reason_codes]},
    )


class Engine:
    def __init__(
        self,
        *,
        store: Store,
        taxonomy: Taxonomy,
        rules: Rules,
        llm: LLMClient,
        settings: Settings,
        dry_run: bool = False,
    ):
        self.store = store
        self.taxonomy = taxonomy
        self.rules = rules
        self.llm = llm
        self.settings = settings
        self.dry_run = dry_run
        self.index = cand_mod.CandidateIndex(taxonomy)
        # Outcomes are stamped with rules + taxonomy content, so a changed category spec is a new
        # version: ingest supersedes, revalidate re-decides from stored candidates (AC-2.4, AC-3.4).
        self.version = pipeline_version(rules.version, taxonomy.content_hash())

    # ------------------------------------------------------------------ batch

    def run_batch(
        self,
        *,
        supplier_id: str | None = None,
        limit: int | None = None,
        on_item: Callable[[ItemOutcome], None] | None = None,
    ) -> BatchSummary:
        run_id = uuid.uuid4().hex[:12]
        summary = BatchSummary(run_id=run_id)
        for row in list(self.store.iter_feed_rows(supplier_id=supplier_id, limit=limit)):
            with obs.item_context(run_id, row.feed_item_id):
                outcome = self.process_item(row, run_id)
                _log_outcome(outcome)
            if outcome.skipped:
                summary.skipped += 1
            else:
                summary.processed += 1
                if outcome.error:
                    summary.errors += 1
                if outcome.status == "draft":
                    summary.draft += 1
                elif outcome.status == "quarantined":
                    summary.quarantined += 1
            if on_item:
                on_item(outcome)
        return summary

    # ------------------------------------------------------------------- item

    def process_item(self, row: RawFeedRow, run_id: str) -> ItemOutcome:
        rv = self.version
        if self.store.processed_status(row.feed_item_id, rv) is not None:
            return ItemOutcome(feed_item_id=row.feed_item_id, status=None, skipped=True)

        trail = _Trail(self.store, run_id, row.feed_item_id)
        stage = "clean"
        try:
            candidate, result = self._run_stages(row, trail)
            status = route.route(result)
            stage = "route"
            t0 = time.perf_counter()
            self._persist(row.feed_item_id, row.supplier_id, candidate, result, status, run_id)
            trail.record("route", None, {"status": status, "codes": [c.code for c in result.reason_codes]}, t0)
            self.store.mark_processed(row.feed_item_id, rv, status, run_id)
            self._commit()
            return ItemOutcome(row.feed_item_id, status, reason_codes=result.reason_codes)
        except Exception as exc:  # noqa: BLE001 - one bad row must not abort the batch
            self.store.rollback()
            stage = getattr(exc, "stage", stage)
            err = f"{type(exc).__name__}: {exc}"
            log.exception("item failed", extra={"event": "item_error", "stage": stage})
            code = ReasonCode(code=f"PIPELINE_ERROR:{stage}", severity="blocking", stage="route", detail=err)
            try:
                trail.seq = 0
                trail.record(
                    "error",
                    None,
                    {"stage": stage, "error": err, "trace": traceback.format_exc()[-2000:]},
                    time.perf_counter(),
                )
                self.store.supersede(row.feed_item_id)
                self.store.insert_quarantine(
                    feed_item_id=row.feed_item_id,
                    supplier_id=row.supplier_id,
                    proposed_category_id=None,
                    candidate=None,
                    reason_codes=[code],
                    run_id=run_id,
                    rules_version=rv,
                )
                self.store.mark_processed(row.feed_item_id, rv, "quarantined", run_id)
                self._commit()
            except Exception:  # noqa: BLE001
                self.store.rollback()
                log.exception("could not record failure", extra={"event": "item_error"})
            return ItemOutcome(row.feed_item_id, "quarantined", error=err, reason_codes=[code])

    # ------------------------------------------------------------- revalidate

    def revalidate_batch(
        self, *, limit: int | None = None, on_item: Callable[[ItemOutcome], None] | None = None
    ) -> BatchSummary:
        """AC-2.4: re-decide every stored outcome under the current rules + taxonomy without calling
        the LLM. Items already decided under this version are skipped (idempotent)."""
        run_id = uuid.uuid4().hex[:12]
        summary = BatchSummary(run_id=run_id)
        for feed_item_id, candidate_json in list(self.store.iter_stored_candidates(limit=limit)):
            if self.store.processed_status(feed_item_id, self.version) is not None:
                summary.skipped += 1
                continue
            with obs.item_context(run_id, feed_item_id):
                outcome = self.revalidate_item(CandidateListing.model_validate_json(candidate_json), run_id)
                _log_outcome(outcome)
            summary.processed += 1
            summary.errors += 1 if outcome.error else 0
            summary.draft += 1 if outcome.status == "draft" else 0
            summary.quarantined += 1 if outcome.status == "quarantined" else 0
            if on_item:
                on_item(outcome)
        return summary

    def revalidate_item(self, candidate: CandidateListing, run_id: str) -> ItemOutcome:
        listing = candidate.normalized
        fid = listing.feed_item_id
        trail = _Trail(self.store, run_id, fid)
        try:
            t0 = time.perf_counter()
            # Upstream decisions (clean, candidates, classification) are kept as they were made.
            codes = [c for c in candidate.reason_codes if c.stage in UPSTREAM_STAGES]
            category_id = candidate.category_id
            if category_id is not None and (category_id not in self.taxonomy or not self.taxonomy.is_leaf(category_id)):
                codes.append(ReasonCode(code="CLASSIFICATION_NOT_LEAF", severity="blocking", stage="classify",
                                        detail=f"{category_id} is no longer a leaf in the taxonomy"))
                category_id = None
            llm_attrs = stored_llm_attrs(candidate.llm_trace)
            trail.record("revalidate", {"rules_version": self.version, "llm_calls": 0},
                         {"category_id": category_id, "llm_attrs": llm_attrs}, t0)
            attrs, extra, decided = self._reconcile_validate(listing, category_id, llm_attrs, trail)
            codes.extend(decided)
            updated = candidate.model_copy(update={
                "category_id": category_id, "attributes": attrs, "extra_attributes": extra,
                "reason_codes": list(codes),
            })
            codes.extend(self._comply(listing, category_id, attrs, trail))
            result = ValidationResult(reason_codes=codes)
            status = route.route(result)
            t0 = time.perf_counter()
            self._persist(fid, listing.supplier_id, updated, result, status, run_id)
            trail.record("route", None, {"status": status, "codes": [c.code for c in codes]}, t0)
            self.store.mark_processed(fid, self.version, status, run_id)
            self._commit()
            return ItemOutcome(fid, status, reason_codes=codes)
        except Exception as exc:  # noqa: BLE001 - one bad stored candidate must not abort the run
            self.store.rollback()
            err = f"{type(exc).__name__}: {exc}"
            log.exception("revalidate failed", extra={"event": "item_error", "stage": "revalidate"})
            return ItemOutcome(fid, None, error=err)

    def _commit(self) -> None:
        if self.dry_run:
            self.store.rollback()
        else:
            self.store.commit()

    # ----------------------------------------------------------------- stages

    def _run_stages(self, row: RawFeedRow, trail: _Trail) -> tuple[CandidateListing, ValidationResult]:
        s = self.settings
        codes: list[ReasonCode] = []

        t0 = time.perf_counter()
        listing = _staged("clean", clean.normalize, row, self.rules)
        trail.record("clean", row, listing, t0)
        codes.extend(listing.reason_codes)

        # candidates (deterministic)
        t0 = time.perf_counter()
        cands = _staged("candidates", self.index.shortlist, listing.search_text, s.candidate_k)
        if not cands and cand_mod.tokenize(listing.search_text):
            # No lexical overlap with any leaf: let the model choose from the full leaf set.
            cands = self.index.all_leaves()
            codes.append(ReasonCode(code="CLASSIFICATION_WEAK_CANDIDATES", severity="warning", stage="candidates"))
        trail.record("candidates", {"text": listing.search_text}, {"candidates": cands}, t0)

        # classify (LLM proposes, Python decides)
        t0 = time.perf_counter()
        category_id, source, llm_cls, cls_trace, cls_codes = _staged("classify", self._classify, listing, cands)
        codes.extend(cls_codes)
        trail.record(
            "classify",
            {"candidates": [c.category_id for c in cands]},
            {"category_id": category_id, "source": source, "llm": llm_cls, "codes": [c.code for c in cls_codes]},
            t0,
            llm_raw=cls_trace,
        )

        ext_trace: list[LLMAttempt] = []
        llm_attrs: dict[str, Any] | None = None
        if category_id is not None:
            missing = reconcile.missing_specs(self.taxonomy.required_attributes(category_id), listing.det_attrs)
            if missing:
                t0 = time.perf_counter()
                parsed, ext_trace = _staged(
                    "classify", runner.extract, self.llm, listing, missing, max_retries=s.llm_max_retries
                )
                if parsed is None:
                    # AC-2.2: exhausted retries quarantine the item; the stage goes in `detail`.
                    codes.append(
                        ReasonCode(code=llm_failure_code(ext_trace), severity="blocking", stage="classify",
                                   detail="extract")
                    )
                else:
                    llm_attrs = parsed.model_dump()
                trail.record(
                    "classify", {"missing": [m.key for m in missing]}, {"llm_attrs": llm_attrs}, t0, llm_raw=ext_trace
                )

        attrs, extra, decided = self._reconcile_validate(listing, category_id, llm_attrs, trail)
        codes.extend(decided)
        candidate = CandidateListing(
            normalized=listing,
            category_id=category_id,
            classification_source=source,
            det_top1=cands[0] if cands else None,
            llm_classification=llm_cls,
            attributes=attrs,
            extra_attributes=extra,
            llm_trace=[*cls_trace, *ext_trace],
            reason_codes=list(codes),
        )
        codes.extend(self._comply(listing, category_id, attrs, trail))
        return candidate, ValidationResult(reason_codes=codes)

    # ------------------------------------------ deterministic decision (shared)

    def _reconcile_validate(
        self,
        listing: NormalizedListing,
        category_id: str | None,
        llm_attrs: dict[str, Any] | None,
        trail: _Trail,
    ) -> tuple[dict[str, AttrValue], dict[str, AttrValue], list[ReasonCode]]:
        """reconcile -> validate. Pure Python; used by ingest and by revalidate (AC-2.4)."""
        if category_id is None:
            return {}, {}, []
        specs = self.taxonomy.required_attributes(category_id)
        t0 = time.perf_counter()
        attrs, extra, rec_codes = _staged("reconcile", reconcile.merge_attributes, specs, listing.det_attrs, llm_attrs)
        trail.record("reconcile", {"det": listing.det_attrs, "llm": llm_attrs}, {"attrs": attrs, "extra": extra}, t0)
        t0 = time.perf_counter()
        attrs, val_codes = _staged("validate", validate.validate_attributes, specs, attrs)
        trail.record("validate", None, {"attrs": attrs, "codes": [c.code for c in val_codes]}, t0)
        return attrs, extra, [*rec_codes, *val_codes]

    def _comply(
        self, listing: NormalizedListing, category_id: str | None, attrs: dict[str, AttrValue], trail: _Trail
    ) -> list[ReasonCode]:
        t0 = time.perf_counter()
        fp = comply.fingerprint(listing.supplier_id, listing.clean_title, attrs)
        comp_codes: list[ReasonCode] = _staged(
            "comply",
            comply.check_compliance,
            clean_title=listing.clean_title,
            scan_text=listing.compliance_text,
            category_prohibited=self.taxonomy.effective_prohibited(category_id) if category_id else (),
            global_prohibited=self.rules.global_prohibited,
            price_min=listing.price_min,
            duplicate_exists=lambda: self.store.fingerprint_exists(fp, exclude_feed_item_id=listing.feed_item_id),
        )
        trail.record("comply", {"fingerprint": fp}, {"codes": [c.code for c in comp_codes]}, t0)
        return comp_codes

    def _classify(
        self, listing: NormalizedListing, cands: list[Candidate]
    ) -> tuple[str | None, str | None, LLMClassification | None, list[LLMAttempt], list[ReasonCode]]:
        codes: list[ReasonCode] = []
        if not cands:
            codes.append(ReasonCode(code="CLASSIFICATION_NO_CANDIDATES", severity="blocking", stage="candidates"))
            return None, None, None, [], codes

        det_top1 = cands[0]
        det_conf = cand_mod.deterministic_confidence(cands)
        llm_cls, trace = runner.classify(self.llm, listing, cands, max_retries=self.settings.llm_max_retries)

        cand_ids = {c.category_id for c in cands}
        if llm_cls is None:
            code = llm_failure_code(trace, candidate_ids=cand_ids)
            codes.append(ReasonCode(code=code, severity="blocking", stage="classify", detail="classify"))
            return det_top1.category_id, "deterministic", None, trace, codes

        if llm_cls.category_id not in cand_ids:  # defence in depth; the Literal model already rejects it
            codes.append(ReasonCode(code="CLASSIFICATION_OUT_OF_SET", severity="blocking", stage="classify"))
            return det_top1.category_id, "deterministic", llm_cls, trace, codes

        if llm_cls.category_id == det_top1.category_id:
            # FR-2a: "agree" only when the deterministic top-1 is itself confident.
            confident = det_conf >= self.settings.high_conf_threshold
            chosen, source = llm_cls.category_id, ("agree" if confident else "llm")
        else:
            severity = (
                "blocking"
                if det_conf >= self.settings.high_conf_threshold
                and llm_cls.confidence < self.settings.disagree_quarantine_below
                else "warning"
            )
            codes.append(
                ReasonCode(
                    code="CLASSIFICATION_DISAGREE",
                    severity=severity,
                    stage="classify",
                    detail=(
                        f"det={det_top1.category_id}({det_conf:.2f}) "
                        f"llm={llm_cls.category_id}({llm_cls.confidence:.2f})"
                    ),
                )
            )
            chosen, source = llm_cls.category_id, "llm"

        if not self.taxonomy.is_leaf(chosen):
            codes.append(
                ReasonCode(code="CLASSIFICATION_NOT_LEAF", severity="blocking", stage="classify", detail=chosen)
            )
        return chosen, source, llm_cls, trace, codes

    # ---------------------------------------------------------------- persist

    def _persist(
        self,
        feed_item_id: str,
        supplier_id: str,
        candidate: CandidateListing,
        result: ValidationResult,
        status: Status,
        run_id: str,
    ) -> None:
        self.store.supersede(feed_item_id)
        listing = candidate.normalized
        if status == "draft":
            if candidate.category_id is None:  # explicit: asserts vanish under python -O
                raise RuntimeError("routing invariant broken: draft listing without category_id")
            self.store.insert_draft(
                feed_item_id=feed_item_id,
                supplier_id=supplier_id,
                category_id=candidate.category_id,
                clean_title=listing.clean_title,
                attributes=candidate.attributes,
                reason_codes=result.reason_codes,
                price=listing.price_min,
                fingerprint=comply.fingerprint(listing.supplier_id, listing.clean_title, candidate.attributes),
                run_id=run_id,
                rules_version=self.version,
                candidate=candidate.model_dump(mode="json"),
            )
        else:
            self.store.insert_quarantine(
                feed_item_id=feed_item_id,
                supplier_id=supplier_id,
                proposed_category_id=candidate.category_id,
                candidate=candidate.model_dump(mode="json"),
                reason_codes=result.reason_codes,
                run_id=run_id,
                rules_version=self.version,
            )


class StageError(Exception):
    def __init__(self, stage: str, cause: BaseException):
        super().__init__(f"{stage}: {type(cause).__name__}: {cause}")
        self.stage = stage
        self.__cause__ = cause


def _staged(stage: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except StageError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise StageError(stage, exc) from exc
