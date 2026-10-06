from __future__ import annotations

import asyncio

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.backends.base import InferenceBackend, ModelResult
from app.core.circuit_breaker import CircuitBreaker, CircuitOpenError
from app.metrics import BACKEND_ERRORS

_TRACER = trace.get_tracer(__name__)


class BackendUnavailableError(RuntimeError):
    pass


class ResilientBackend(InferenceBackend):
    def __init__(
        self,
        primary: InferenceBackend,
        fallback: InferenceBackend,
        breaker: CircuitBreaker,
        *,
        primary_timeout_ms: int,
        fallback_timeout_ms: int,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.breaker = breaker
        self.primary_timeout_s = primary_timeout_ms / 1000
        self.fallback_timeout_s = fallback_timeout_ms / 1000
        self.fallback_requests = 0
        self.primary_failures = 0
        self.primary_timeouts = 0
        self.fallback_failures = 0

    async def infer_batch(self, texts: list[str]) -> list[ModelResult]:
        try:
            self.breaker.before_call()
        except CircuitOpenError:
            return await self._serve_fallback(texts)

        try:
            with _TRACER.start_as_current_span("inference.primary") as span:
                span.set_attribute("inference.batch.size", len(texts))
                span.set_attribute("inference.backend.timeout_ms", self.primary_timeout_s * 1000)
                try:
                    results = await asyncio.wait_for(
                        self.primary.infer_batch(texts),
                        timeout=self.primary_timeout_s,
                    )
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR))
                    raise
        except TimeoutError:
            self.primary_timeouts += len(texts)
            BACKEND_ERRORS.labels(stage="primary", reason="timeout").inc(len(texts))
            self.breaker.record_failure()
            return await self._serve_fallback(texts)
        except Exception:
            self.primary_failures += len(texts)
            BACKEND_ERRORS.labels(stage="primary", reason="error").inc(len(texts))
            self.breaker.record_failure()
            return await self._serve_fallback(texts)

        self.breaker.record_success()
        return results

    async def _serve_fallback(self, texts: list[str]) -> list[ModelResult]:
        try:
            with _TRACER.start_as_current_span("inference.fallback") as span:
                span.set_attribute("inference.batch.size", len(texts))
                span.set_attribute("inference.backend.timeout_ms", self.fallback_timeout_s * 1000)
                try:
                    fallback = await asyncio.wait_for(
                        self.fallback.infer_batch(texts),
                        timeout=self.fallback_timeout_s,
                    )
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR))
                    raise
        except Exception as exc:
            self.fallback_failures += len(texts)
            BACKEND_ERRORS.labels(stage="fallback", reason="error_or_timeout").inc(len(texts))
            raise BackendUnavailableError(
                "primary and fallback inference backends unavailable"
            ) from exc

        self.fallback_requests += len(texts)
        return [
            ModelResult(
                label=item.label,
                score=item.score,
                backend=item.backend,
                fallback_used=True,
            )
            for item in fallback
        ]

    async def aclose(self) -> None:
        await self.primary.aclose()
        await self.fallback.aclose()
