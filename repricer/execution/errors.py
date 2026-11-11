from __future__ import annotations

from repricer.models import ReasonCode


class ExecutionError(Exception):
    code = "EXECUTION_ERROR"


class IllegalTransition(ExecutionError):
    code = "ILLEGAL_TRANSITION"


class StaleSnapshot(ExecutionError):
    code = "STALE_SNAPSHOT"


class FloorViolation(ExecutionError):
    code = "FLOOR_VIOLATION"


class GuardrailViolation(ExecutionError):
    code = "GUARDRAIL_VIOLATION"

    def __init__(self, reason: ReasonCode, message: str) -> None:
        super().__init__(message)
        self.reason = reason
