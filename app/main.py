from __future__ import annotations

import asyncio
import statistics
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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
from app.flight_recorder import FlightRecorder
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
from app.models import (
    BuildInfo,
    PolicyLabRequest,
    PredictRequest,
    PredictResponse,
    RuntimeStatus,
)
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
    app.state.flight_recorder = FlightRecorder()
    app.state.shadow_stable = MockBackend(
        "stable-v17",
        base_latency_ms=28,
        item_latency_ms=4,
        score_salt="stable-v17:",
    )
    app.state.shadow_candidate = MockBackend(
        "candidate-v18",
        base_latency_ms=20,
        item_latency_ms=4,
        score_salt="candidate-v18:",
    )
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
    recorder: FlightRecorder = request.app.state.flight_recorder
    request_id: str = request.state.request_id
    recorder.start(request_id, payload.text)
    recorder.record(request_id, "admitted", {"backend_mode": settings.backend_mode})
    started = time.perf_counter()
    try:
        outcome = await batcher.submit(
            payload.text,
            timeout_ms=settings.request_timeout_ms,
            request_id=request_id,
            event_hook=recorder.record,
        )
    except QueueFullError as exc:
        recorder.finish(request_id, "overloaded")
        REQUESTS.labels(status="overloaded", backend="none", fallback="false").inc()
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except TimeoutError as exc:
        recorder.finish(request_id, "timeout")
        DEADLINE_EXPIRED_TOTAL.inc()
        REQUESTS.labels(status="timeout", backend="none", fallback="false").inc()
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="inference timeout",
        ) from exc
    except BackendUnavailableError as exc:
        recorder.finish(request_id, "unavailable")
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
    recorder.finish(request_id, "ok")

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


@app.post("/v1/shadow-compare")
async def shadow_compare(payload: PredictRequest, request: Request) -> dict[str, Any]:
    stable: InferenceBackend = request.app.state.shadow_stable
    candidate: InferenceBackend = request.app.state.shadow_candidate

    async def timed(backend: InferenceBackend) -> dict[str, Any]:
        started = time.perf_counter()
        result = (await backend.infer_batch([payload.text]))[0]
        return {
            "backend": result.backend,
            "label": result.label,
            "score": result.score,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    stable_result, candidate_result = await asyncio.gather(
        timed(stable),
        timed(candidate),
    )
    agreement = stable_result["label"] == candidate_result["label"]
    score_delta = abs(stable_result["score"] - candidate_result["score"])
    gate = "eligible_for_next_stage"
    if not agreement or score_delta > 0.25:
        gate = "hold_candidate"

    return {
        "mode": "shadow_only",
        "serving_decision": "stable-v17",
        "stable": stable_result,
        "candidate": candidate_result,
        "agreement": agreement,
        "score_delta": round(score_delta, 6),
        "gate": gate,
        "note": "synthetic comparison; candidate never changes the live response",
    }


@app.post("/v1/policy-lab")
async def policy_lab(payload: PolicyLabRequest) -> dict[str, Any]:
    traffic = {
        "steady": {"requests": 24, "concurrency": 8, "failure_every": 0},
        "flash_crowd": {"requests": 48, "concurrency": 32, "failure_every": 0},
        "degraded_primary": {"requests": 36, "concurrency": 18, "failure_every": 6},
    }[payload.profile]
    policies = {
        "latency_guard": {
            "queue_capacity": 8,
            "batch_size": 4,
            "batch_wait_ms": 4,
            "timeout_ms": 220,
        },
        "throughput_guard": {
            "queue_capacity": 32,
            "batch_size": 8,
            "batch_wait_ms": 12,
            "timeout_ms": 500,
        },
    }
    results = {}
    for name, policy in policies.items():
        results[name] = await _run_policy_trial(
            name=name,
            policy=policy,
            traffic=traffic,
        )

    latency_guard = results["latency_guard"]
    throughput_guard = results["throughput_guard"]
    if abs(latency_guard["success_rate"] - throughput_guard["success_rate"]) > 0.05:
        winner = max(results, key=lambda key: results[key]["success_rate"])
        reason = "higher admission success under this traffic profile"
    else:
        winner = min(results, key=lambda key: results[key]["p95_ms"])
        reason = "lower p95 with comparable success rate"

    return {
        "profile": payload.profile,
        "traffic": traffic,
        "policies": results,
        "winner_for_this_run": winner,
        "reason": reason,
        "note": "isolated synthetic trial using the same batcher and resilience code path",
    }


async def _run_policy_trial(
    *,
    name: str,
    policy: dict[str, int],
    traffic: dict[str, int],
) -> dict[str, Any]:
    breaker = CircuitBreaker(failure_threshold=3, recovery_seconds=0.08)
    primary = MockBackend(
        f"{name}-primary",
        base_latency_ms=32,
        item_latency_ms=6,
        fail_on_marker=True,
        score_salt=f"{name}:",
    )
    fallback = MockBackend(
        f"{name}-fallback",
        base_latency_ms=12,
        item_latency_ms=2,
        score_salt=f"{name}-fallback:",
    )
    backend = ResilientBackend(
        primary,
        fallback,
        breaker,
        primary_timeout_ms=max(80, policy["timeout_ms"] - 40),
        fallback_timeout_ms=120,
    )
    batcher = DynamicBatcher(
        backend,
        max_batch_size=policy["batch_size"],
        max_wait_ms=policy["batch_wait_ms"],
        capacity=policy["queue_capacity"],
        drain_timeout_ms=1000,
    )
    await batcher.start()
    semaphore = asyncio.Semaphore(traffic["concurrency"])
    statuses: Counter[str] = Counter()
    latencies: list[float] = []
    batch_sizes: list[int] = []
    fallback_count = 0

    async def one(index: int) -> None:
        nonlocal fallback_count
        text = f"policy-lab-{index}"
        failure_every = traffic["failure_every"]
        if failure_every and (index + 1) % failure_every == 0:
            text += " __FAIL_PRIMARY__"
        async with semaphore:
            started = time.perf_counter()
            try:
                outcome = await batcher.submit(text, timeout_ms=policy["timeout_ms"])
                statuses["ok"] += 1
                batch_sizes.append(outcome.batch_size)
                fallback_count += int(outcome.result.fallback_used)
            except QueueFullError:
                statuses["overloaded"] += 1
            except TimeoutError:
                statuses["timeout"] += 1
            except BackendUnavailableError:
                statuses["unavailable"] += 1
            latencies.append((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    try:
        await asyncio.gather(*(one(index) for index in range(traffic["requests"])))
    finally:
        await batcher.stop()
    duration = time.perf_counter() - started
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, round((len(ordered) - 1) * 0.95))

    return {
        "policy": policy,
        "status_counts": dict(statuses),
        "success_rate": round(statuses["ok"] / traffic["requests"], 4),
        "throughput_rps": round(traffic["requests"] / duration, 2),
        "p95_ms": round(ordered[p95_index], 2),
        "mean_batch_size": round(statistics.mean(batch_sizes), 2)
        if batch_sizes
        else 0,
        "fallback_requests": fallback_count,
        "breaker_state": breaker.state.value,
    }


@app.get("/ops/flights")
async def recent_flights(request: Request, limit: int = 12) -> dict[str, Any]:
    recorder: FlightRecorder = request.app.state.flight_recorder
    return {"flights": recorder.recent(limit)}


@app.get("/ops/flights/{request_id}")
async def flight_detail(request_id: str, request: Request) -> dict[str, Any]:
    recorder: FlightRecorder = request.app.state.flight_recorder
    flight = recorder.get(request_id)
    if flight is None:
        raise HTTPException(status_code=404, detail="request flight not found")
    return flight


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
