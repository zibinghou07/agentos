from __future__ import annotations

import asyncio
import inspect
import random
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

import httpx
from jsonschema import ValidationError


class ErrorType(StrEnum):
    TIMEOUT = "TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"
    SERVER_ERROR = "SERVER_ERROR"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    EMPTY_RESULT = "EMPTY_RESULT"
    DEPENDENCY_FAILED = "DEPENDENCY_FAILED"
    LOGIC_FAILURE = "LOGIC_FAILURE"
    AMBIGUOUS_OPERATION = "AMBIGUOUS_OPERATION"
    AUTH_ERROR = "AUTH_ERROR"
    UNKNOWN = "UNKNOWN"


class EmptyResultError(RuntimeError):
    pass


class ToolCapabilityError(RuntimeError):
    pass


class VerificationError(RuntimeError):
    def __init__(self, failures: list[str], action: str):
        super().__init__("verification failed: " + ", ".join(failures))
        self.failures = failures
        self.action = action


class AmbiguousOperationError(RuntimeError):
    pass


class RecoveryExhausted(RuntimeError):
    def __init__(self, error_type: ErrorType, action: str, cause: Exception):
        super().__init__(f"{error_type}: {cause}")
        self.error_type = error_type
        self.action = action
        self.cause = cause


def classify_error(exc: Exception) -> ErrorType:
    text = str(exc).lower()
    if isinstance(exc, AmbiguousOperationError):
        return ErrorType.AMBIGUOUS_OPERATION
    if isinstance(exc, EmptyResultError) or "empty result" in text:
        return ErrorType.EMPTY_RESULT
    if isinstance(exc, VerificationError):
        return ErrorType.LOGIC_FAILURE
    if isinstance(exc, ToolCapabilityError):
        return ErrorType.LOGIC_FAILURE
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        return ErrorType.TIMEOUT
    if isinstance(exc, httpx.TransportError):
        return ErrorType.SERVER_ERROR
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in {401, 403}:
            return ErrorType.AUTH_ERROR
        if status == 429:
            return ErrorType.RATE_LIMIT
        if status >= 500:
            return ErrorType.SERVER_ERROR
    if "unauthorized" in text or "invalid api key" in text or "forbidden" in text:
        return ErrorType.AUTH_ERROR
    if "rate" in text and "limit" in text:
        return ErrorType.RATE_LIMIT
    if "server error" in text or "service unavailable" in text:
        return ErrorType.SERVER_ERROR
    if isinstance(exc, ValidationError) or "schema" in text or "validation" in text:
        return ErrorType.INVALID_SCHEMA
    if (
        isinstance(exc, (TypeError, ValueError, KeyError))
        or "argument" in text
        or "not found" in text
    ):
        return ErrorType.INVALID_ARGUMENT
    return ErrorType.UNKNOWN


def escalation_action(kind: ErrorType) -> str:
    """What the runtime should do once node-local recovery is exhausted."""
    if kind == ErrorType.AMBIGUOUS_OPERATION:
        return "manual"
    if kind == ErrorType.AUTH_ERROR:
        # A new plan cannot fix a credential problem; fail fast instead of spending tokens.
        return "fail"
    return "replan"


Repair = Callable[
    [dict[str, Any], Exception], dict[str, Any] | Awaitable[dict[str, Any]]
]


class RecoveryManager:
    def __init__(
        self,
        retries: int = 2,
        repairs: int = 1,
        replans: int = 1,
        *,
        base_delay: float = 0.25,
        max_delay: float = 4.0,
        jitter: float = 0.2,
        seed: int = 42,
    ):
        self.retries, self.repairs, self.replans = retries, repairs, replans
        self.base_delay, self.max_delay, self.jitter = base_delay, max_delay, jitter
        self.random = random.Random(seed)

    async def execute(
        self,
        call: Callable[[dict[str, Any]], Awaitable[Any]],
        arguments: dict[str, Any],
        repair: Repair | None = None,
        allow_empty_list: bool = False,
    ) -> tuple[Any, list[str]]:
        trace: list[str] = []
        retry_count = repair_count = 0
        current = dict(arguments)
        while True:
            try:
                result = await call(current)
                # An empty list is a real answer ("no matching orders") for list-shaped tools.
                if result in (None, "", {}) or (result == [] and not allow_empty_list):
                    raise EmptyResultError("empty result")
                return result, trace
            except Exception as exc:
                kind = classify_error(exc)
                transient = {
                    ErrorType.TIMEOUT,
                    ErrorType.RATE_LIMIT,
                    ErrorType.SERVER_ERROR,
                }
                if kind in transient and retry_count < self.retries:
                    retry_count += 1
                    delay = min(self.max_delay, self.base_delay * 2 ** (retry_count - 1))
                    delay *= 1 + self.random.uniform(-self.jitter, self.jitter)
                    trace.append(f"RETRY:{kind}:{retry_count}:delay={delay:.3f}")
                    await asyncio.sleep(max(0, delay))
                    continue
                repairable = {ErrorType.INVALID_ARGUMENT, ErrorType.INVALID_SCHEMA}
                if kind in repairable and repair and repair_count < self.repairs:
                    repair_count += 1
                    repaired = repair(current, exc)
                    current = await repaired if inspect.isawaitable(repaired) else repaired
                    trace.append(f"REPAIR:{kind}:{repair_count}")
                    continue
                raise RecoveryExhausted(kind, escalation_action(kind), exc) from exc
