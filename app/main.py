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
from app.core.slo_governor import choose_serving_policy
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
    SLODecisionRequest,
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


def _traffic_profile(profile: str) -> dict[str, int]:
    return {
        "steady": {"requests": 24, "concurrency": 8, "failure_every": 0},
        "flash_crowd": {"requests": 48, "concurrency": 32, "failure_every": 0},
        "degraded_primary": {"requests": 36, "concurrency": 18, "failure_every": 6},
    }[profile]


def _candidate_policies() -> dict[str, dict[str, int]]:
    return {
        "latency_guard": {
            "queue_capacity": 8,
            "batch_size": 4,
            "batch_wait_ms": 4,
            "timeout_ms": 220,
            "primary_timeout_ms": 180,
            "breaker_threshold": 3,
        },
        "throughput_guard": {
            "queue_capacity": 32,
            "batch_size": 8,
            "batch_wait_ms": 12,
            "timeout_ms": 500,
            "primary_timeout_ms": 460,
            "breaker_threshold": 3,
        },
        "availability_guard": {
            "queue_capacity": 16,
            "batch_size": 4,
            "batch_wait_ms": 5,
            "timeout_ms": 320,
            "primary_timeout_ms": 80,
            "breaker_threshold": 2,
        },
    }


@app.post("/v1/policy-lab")
async def policy_lab(payload: PolicyLabRequest) -> dict[str, Any]:
    traffic = _traffic_profile(payload.profile)
    policies = _candidate_policies()
    results = {}
    for name, policy in policies.items():
        results[name] = await _run_policy_trial(
            name=name,
            policy=policy,
            traffic=traffic,
        )

    winner = max(
        results,
        key=lambda key: (
            results[key]["success_rate"],
            -results[key]["p95_ms"],
            -results[key]["throughput_rps"],
        ),
    )
    return {
        "profile": payload.profile,
        "traffic": traffic,
        "policies": results,
        "winner_for_this_run": winner,
        "reason": "highest success rate, then lower p95 and higher admitted throughput",
        "note": "isolated synthetic trial using the same batcher and resilience code path",
    }


@app.post("/v1/slo-decision")
async def slo_decision(payload: SLODecisionRequest) -> dict[str, Any]:
    traffic = _traffic_profile(payload.profile)
    results = {}
    for name, policy in _candidate_policies().items():
        results[name] = await _run_policy_trial(
            name=name,
            policy=policy,
            traffic=traffic,
        )
    receipt = choose_serving_policy(
        profile=payload.profile,
        traffic=traffic,
        results=results,
        target_p95_ms=payload.target_p95_ms,
        min_success_rate=payload.min_success_rate,
    )
    envelope = await _measure_safe_envelope(
        policy_name=receipt["selected_policy"],
        policy=receipt["selected_config"],
        failure_every=traffic["failure_every"],
        target_p95_ms=payload.target_p95_ms,
        min_success_rate=payload.min_success_rate,
    )
    receipt["safe_operating_envelope"] = envelope
    receipt["deployment_contract"] = {
        "runtime_config": receipt["selected_config"],
        "max_verified_concurrency": envelope["max_verified_concurrency"],
        "first_unsafe_concurrency": envelope["first_unsafe_concurrency"],
        "scale_before_concurrency": envelope["scale_before_concurrency"],
        "shed_from_concurrency": envelope["first_unsafe_concurrency"],
        "evidence_basis": "measured concurrency sweep on isolated serving path",
    }
    return {
        "profile": payload.profile,
        "traffic": traffic,
        "decision_receipt": receipt,
        "note": (
            "same synthetic workload replayed through isolated copies of the real "
            "batcher and resilience path"
        ),
    }


async def _run_policy_trial(
    *,
    name: str,
    policy: dict[str, int],
    traffic: dict[str, int],
) -> dict[str, Any]:
    breaker = CircuitBreaker(
        failure_threshold=policy["breaker_threshold"],
        recovery_seconds=0.08,
    )
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
        primary_timeout_ms=policy["primary_timeout_ms"],
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
    success_latencies: list[float] = []
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
            succeeded = False
            try:
                outcome = await batcher.submit(text, timeout_ms=policy["timeout_ms"])
                statuses["ok"] += 1
                succeeded = True
                batch_sizes.append(outcome.batch_size)
                fallback_count += int(outcome.result.fallback_used)
            except QueueFullError:
                statuses["overloaded"] += 1
            except TimeoutError:
                statuses["timeout"] += 1
            except BackendUnavailableError:
                statuses["unavailable"] += 1
            elapsed_ms = (time.perf_counter() - started) * 1000
            latencies.append(elapsed_ms)
            if succeeded:
                success_latencies.append(elapsed_ms)

    started = time.perf_counter()
    try:
        await asyncio.gather(*(one(index) for index in range(traffic["requests"])))
    finally:
        await batcher.stop()
    duration = time.perf_counter() - started
    ordered = sorted(latencies)
    success_ordered = sorted(success_latencies)
    p95_source = success_ordered or ordered
    p95_index = min(len(p95_source) - 1, round((len(p95_source) - 1) * 0.95))

    backend_calls = primary.batch_calls + fallback.batch_calls
    successful = statuses["ok"]
    calls_per_100 = round(backend_calls / successful * 100, 2) if successful else 9999.0

    return {
        "policy": policy,
        "status_counts": dict(statuses),
        "success_rate": round(successful / traffic["requests"], 4),
        "throughput_rps": round(successful / duration, 2),
        "offered_rps": round(traffic["requests"] / duration, 2),
        "p95_ms": round(p95_source[p95_index], 2),
        "shed_requests": traffic["requests"] - successful,
        "mean_batch_size": round(statistics.mean(batch_sizes), 2)
        if batch_sizes
        else 0,
        "fallback_requests": fallback_count,
        "backend_calls": backend_calls,
        "backend_calls_per_100_success": calls_per_100,
        "breaker_state": breaker.state.value,
    }


async def _measure_safe_envelope(
    *,
    policy_name: str,
    policy: dict[str, int],
    failure_every: int,
    target_p95_ms: float,
    min_success_rate: float,
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    concurrency_levels = [4, 8, 16, 24, 32, 40]

    for concurrency in concurrency_levels:
        trial = await _run_policy_trial(
            name=f"{policy_name}-envelope-{concurrency}",
            policy=policy,
            traffic={
                "requests": max(24, concurrency * 2),
                "concurrency": concurrency,
                "failure_every": failure_every,
            },
        )
        slo_pass = (
            trial["success_rate"] >= min_success_rate
            and trial["p95_ms"] <= target_p95_ms
        )
        points.append(
            {
                "concurrency": concurrency,
                "requests": max(24, concurrency * 2),
                "slo_pass": slo_pass,
                "success_rate": trial["success_rate"],
                "p95_ms": trial["p95_ms"],
                "admitted_rps": trial["throughput_rps"],
                "shed_requests": trial["shed_requests"],
                "backend_calls_per_100_success": trial[
                    "backend_calls_per_100_success"
                ],
            }
        )

    safe_points = [point for point in points if point["slo_pass"]]
    max_safe = max(
        (point["concurrency"] for point in safe_points),
        default=None,
    )
    first_unsafe = next(
        (
            point["concurrency"]
            for point in points
            if not point["slo_pass"]
            and (max_safe is None or point["concurrency"] > max_safe)
        ),
        None,
    )
    if first_unsafe is None:
        scale_before = max_safe
    else:
        lower_safe = [
            point["concurrency"]
            for point in safe_points
            if point["concurrency"] < first_unsafe
        ]
        scale_before = max(lower_safe, default=None)

    return {
        "method": "measured_concurrency_sweep",
        "concurrency_levels": concurrency_levels,
        "points": points,
        "max_verified_concurrency": max_safe,
        "first_unsafe_concurrency": first_unsafe,
        "scale_before_concurrency": scale_before,
        "target_p95_ms": target_p95_ms,
        "min_success_rate": min_success_rate,
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
