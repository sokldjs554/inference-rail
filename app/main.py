from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.backends.base import InferenceBackend
from app.backends.mock import MockBackend
from app.backends.resilient import BackendUnavailableError, ResilientBackend
from app.backends.triton import TritonHTTPBackend
from app.config import settings
from app.core.batcher import DynamicBatcher, QueueFullError
from app.core.circuit_breaker import CircuitBreaker
from app.metrics import (
    BATCH_SIZE,
    CIRCUIT_STATE,
    DEADLINE_EXPIRED_TOTAL,
    FALLBACK_TOTAL,
    QUEUE_CAPACITY,
    QUEUE_DEPTH,
    QUEUE_LATENCY,
    REQUEST_LATENCY,
    REQUESTS,
)
from app.models import BuildInfo, PredictRequest, PredictResponse, RuntimeStatus
from app.telemetry import configure_telemetry


def _build_primary_backend() -> InferenceBackend:
    if settings.backend_mode == "mock":
        return MockBackend(
            "primary-model",
            base_latency_ms=settings.mock_base_latency_ms,
            item_latency_ms=settings.mock_item_latency_ms,
            fail_on_marker=True,
        )
    if settings.backend_mode == "triton":
        return TritonHTTPBackend(
            base_url=settings.triton_url,
            model=settings.triton_model,
            input_name=settings.triton_input_name,
            label_output=settings.triton_label_output,
            score_output=settings.triton_score_output,
        )
    raise RuntimeError(
        f"unsupported BACKEND_MODE={settings.backend_mode!r}; expected mock or triton"
    )


def _build_runtime() -> tuple[CircuitBreaker, ResilientBackend, DynamicBatcher]:
    breaker = CircuitBreaker(
        failure_threshold=settings.breaker_failure_threshold,
        recovery_seconds=settings.breaker_recovery_seconds,
    )
    primary = _build_primary_backend()
    fallback = MockBackend(
        "fallback-model",
        base_latency_ms=max(8.0, settings.mock_base_latency_ms * 0.45),
        item_latency_ms=max(2.0, settings.mock_item_latency_ms * 0.4),
    )
    backend = ResilientBackend(
        primary,
        fallback,
        breaker,
        primary_timeout_ms=settings.primary_timeout_ms,
        fallback_timeout_ms=settings.fallback_timeout_ms,
    )
    batcher = DynamicBatcher(
        backend,
        max_batch_size=settings.max_batch_size,
        max_wait_ms=settings.max_batch_wait_ms,
        capacity=settings.queue_capacity,
        drain_timeout_ms=settings.shutdown_drain_timeout_ms,
    )
    return breaker, backend, batcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    breaker, backend, batcher = _build_runtime()
    app.state.breaker = breaker
    app.state.backend = backend
    app.state.batcher = batcher
    QUEUE_CAPACITY.set(batcher.capacity)
    await batcher.start()
    try:
        yield
    finally:
        await batcher.stop()


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
configure_telemetry(app, settings.otlp_endpoint)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    response.headers["x-response-time-ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    return response


@app.get("/", include_in_schema=False)
async def demo() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/v1/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    batcher: DynamicBatcher = request.app.state.batcher
    request_id: str = request.state.request_id
    started = time.perf_counter()
    try:
        outcome = await batcher.submit(payload.text, timeout_ms=settings.request_timeout_ms)
    except QueueFullError as exc:
        REQUESTS.labels(status="overloaded", backend="none", fallback="false").inc()
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except TimeoutError as exc:
        DEADLINE_EXPIRED_TOTAL.inc()
        REQUESTS.labels(status="timeout", backend="none", fallback="false").inc()
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="inference timeout",
        ) from exc
    except BackendUnavailableError as exc:
        REQUESTS.labels(status="unavailable", backend="none", fallback="false").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="inference backends unavailable",
        ) from exc

    elapsed = time.perf_counter() - started
    REQUEST_LATENCY.observe(elapsed)
    QUEUE_LATENCY.observe(outcome.queue_ms / 1000)
    BATCH_SIZE.observe(outcome.batch_size)
    QUEUE_DEPTH.set(batcher.queue_depth)
    fallback_label = str(outcome.result.fallback_used).lower()
    REQUESTS.labels(status="ok", backend=outcome.result.backend, fallback=fallback_label).inc()
    if outcome.result.fallback_used:
        FALLBACK_TOTAL.inc()

    return PredictResponse(
        request_id=request_id,
        label=outcome.result.label,
        score=outcome.result.score,
        backend=outcome.result.backend,
        fallback_used=outcome.result.fallback_used,
        batch_size=outcome.batch_size,
        queue_ms=round(outcome.queue_ms, 2),
        service_ms=round(outcome.service_ms, 2),
    )


@app.get("/ops/build", response_model=BuildInfo)
async def build_info() -> BuildInfo:
    return BuildInfo(
        app=settings.app_name,
        version=settings.app_version,
        revision=settings.revision,
        environment=settings.environment,
        backend_mode=settings.backend_mode,
    )


@app.get("/ops/status", response_model=RuntimeStatus)
async def runtime_status(request: Request) -> RuntimeStatus:
    batcher: DynamicBatcher = request.app.state.batcher
    breaker: CircuitBreaker = request.app.state.breaker
    backend: ResilientBackend = request.app.state.backend
    return RuntimeStatus(
        backend_mode=settings.backend_mode,
        queue_depth=batcher.queue_depth,
        queue_capacity=batcher.capacity,
        breaker_state=breaker.state.value,
        breaker_failures=breaker.failures,
        processed_requests=batcher.processed_requests,
        expired_requests=batcher.expired_requests,
        fallback_requests=backend.fallback_requests,
        primary_failures=backend.primary_failures,
        primary_timeouts=backend.primary_timeouts,
        fallback_failures=backend.fallback_failures,
    )


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready(request: Request) -> dict[str, str | int]:
    batcher: DynamicBatcher = request.app.state.batcher
    if not batcher.is_running:
        raise HTTPException(status_code=503, detail="batcher not running")
    return {"status": "ready", "queue_depth": batcher.queue_depth}


@app.get("/metrics")
async def metrics(request: Request) -> Response:
    batcher: DynamicBatcher = request.app.state.batcher
    QUEUE_DEPTH.set(batcher.queue_depth)
    QUEUE_CAPACITY.set(batcher.capacity)
    breaker: CircuitBreaker = request.app.state.breaker
    for state in ("closed", "open", "half_open"):
        CIRCUIT_STATE.labels(state=state).set(1 if breaker.state.value == state else 0)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
