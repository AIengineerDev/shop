"""One error envelope for everything: {"error": {"code", "message", "details"}}."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from repricer.db.repo import ConcurrentUpdate, NotFound
from repricer.engine.policy import PolicyError
from repricer.execution.errors import (
    FloorViolation,
    GuardrailViolation,
    IllegalTransition,
    StaleSnapshot,
)
from repricer.money import MoneyError


def envelope(status: int, code: str, message: str, details: Any = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "details": details}},
    )


def install(app: FastAPI) -> None:
    @app.exception_handler(NotFound)
    async def _nf(_: Request, e: NotFound) -> JSONResponse:
        return envelope(404, "NOT_FOUND", str(e))

    @app.exception_handler(IllegalTransition)
    async def _it(_: Request, e: IllegalTransition) -> JSONResponse:
        return envelope(409, e.code, str(e))

    @app.exception_handler(StaleSnapshot)
    async def _ss(_: Request, e: StaleSnapshot) -> JSONResponse:
        return envelope(409, e.code, str(e))

    @app.exception_handler(ConcurrentUpdate)
    async def _cu(_: Request, e: ConcurrentUpdate) -> JSONResponse:
        return envelope(409, "CONCURRENT_UPDATE", str(e))

    @app.exception_handler(FloorViolation)
    async def _fv(_: Request, e: FloorViolation) -> JSONResponse:
        return envelope(422, e.code, str(e))

    @app.exception_handler(GuardrailViolation)
    async def _gv(_: Request, e: GuardrailViolation) -> JSONResponse:
        return envelope(422, e.code, str(e), {"reason": e.reason.value})

    @app.exception_handler(PolicyError)
    async def _pe(_: Request, e: PolicyError) -> JSONResponse:
        return envelope(422, "POLICY_INVALID", str(e))

    @app.exception_handler(MoneyError)
    async def _me(_: Request, e: MoneyError) -> JSONResponse:
        return envelope(422, "MONEY_INVALID", str(e))

    @app.exception_handler(RequestValidationError)
    async def _rv(_: Request, e: RequestValidationError) -> JSONResponse:
        return envelope(422, "VALIDATION_ERROR", "request validation failed", e.errors())
