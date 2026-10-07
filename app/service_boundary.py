from __future__ import annotations

import asyncio
import statistics
import time
from collections import Counter
from typing import Any

from app.backends.mock import MockBackend
from app.backends.resilient import BackendUnavailableError, ResilientBackend
from app.core.batcher import DynamicBatcher, QueueFullError
from app.core.circuit_breaker import CircuitBreaker


async def run_service_boundary_proof(
    *,
    policy_name: str,
    policy: dict[str, int],
    target_p95_ms: float,
    min_success_rate: float,
) -> dict[str, Any]:
    healthy = await _load_scenario(
        name="healthy",
        policy=policy,
        requests=24,
        concurrency=8,
        primary_base_ms=32,
        primary_item_ms=6,
    )
    healthy["passed"] = _slo_pass(healthy, target_p95_ms, min_success_rate)
    healthy["expected_control"] = "normal serving path stays inside SLO"

    flash = await _load_scenario(
        name="flash-crowd",
        policy=policy,
        requests=64,
        concurrency=40,
        primary_base_ms=32,
        primary_item_ms=6,
    )
    flash["passed"] = (
        flash["status_counts"].get("timeout", 0) == 0
        and flash["status_counts"].get("unavailable", 0) == 0
        and flash["shed_requests"] > 0
        and flash["p95_ms"] <= target_p95_ms
    )
    flash["expected_control"] = (
        "shed excess load before hidden queue growth or request timeout"
    )

    slow_primary = await _load_scenario(
        name="slow-primary",
        policy=policy,
        requests=24,
        concurrency=8,
        primary_base_ms=max(policy["primary_timeout_ms"] + 120, 560),
        primary_item_ms=10,
    )
    slow_primary["passed"] = _slo_pass(
        slow_primary,
        target_p95_ms,
        min_success_rate,
    )
    slow_primary["expected_control"] = (
        "slow-but-alive primary must not consume the whole client latency budget"
    )

    patched_policy = dict(policy)
    patch_applied = not slow_primary["passed"]
    if patch_applied:
        patched_policy["primary_timeout_ms"] = min(
            policy["primary_timeout_ms"],
            max(60, int(target_p95_ms * 0.55)),
        )
    hardened_slow_primary = await _load_scenario(
        name="slow-primary-hardened",
        policy=patched_policy,
        requests=24,
        concurrency=8,
        primary_base_ms=max(policy["primary_timeout_ms"] + 120, 560),
        primary_item_ms=10,
    )
    hardened_slow_primary["passed"] = _slo_pass(
        hardened_slow_primary,
        target_p95_ms,
        min_success_rate,
    )
    hardened_slow_primary["expected_control"] = "fallback inside the service SLO budget"

    deadline_waste = await _deadline_waste_scenario()
    failure_recovery = await _failure_recovery_scenario(patched_policy)

    candidate_safe = all(
        item["passed"]
        for item in (healthy, flash, slow_primary, deadline_waste, failure_recovery)
    )
    hardened_safe = all(
        item["passed"]
        for item in (
            healthy,
            flash,
            hardened_slow_primary,
            deadline_waste,
            failure_recovery,
        )
    )

    return {
        "scope": {
            "model_level": (
                "batch/concurrency/instance/latency/throughput profiling belongs to "
                "the model-runtime layer"
            ),
            "service_level": (
                "deadline, admission, queue, load shedding, degraded primary, "
                "fallback and recovery"
            ),
        },
        "candidate_policy": policy_name,
        "candidate_config": policy,
        "candidate_service_safe": candidate_safe,
        "scenarios": {
            "healthy": healthy,
            "flash_crowd": flash,
            "slow_primary": slow_primary,
            "deadline_waste": deadline_waste,
            "failure_recovery": failure_recovery,
        },
        "hardening_patch": {
            "applied": patch_applied,
            "reason": (
                "candidate primary timeout consumed too much of the end-to-end SLO"
                if patch_applied
                else "candidate already passed the slow-primary service boundary"
            ),
            "before_primary_timeout_ms": policy["primary_timeout_ms"],
            "after_primary_timeout_ms": patched_policy["primary_timeout_ms"],
        },
        "hardened_slow_primary": hardened_slow_primary,
        "hardened_service_safe": hardened_safe,
        "service_safe_contract": patched_policy,
        "proof_rule": (
            "model-level PASS is not treated as service-level PASS until every "
            "request-lifecycle boundary is exercised"
        ),
    }


def _slo_pass(
    result: dict[str, Any],
    target_p95_ms: float,
    min_success_rate: float,
) -> bool:
    return (
        result["success_rate"] >= min_success_rate
        and result["p95_ms"] <= target_p95_ms
    )


async def _load_scenario(
    *,
    name: str,
    policy: dict[str, int],
    requests: int,
    concurrency: int,
    primary_base_ms: float,
    primary_item_ms: float,
) -> dict[str, Any]:
    breaker = CircuitBreaker(
        failure_threshold=policy["breaker_threshold"],
        recovery_seconds=0.08,
    )
    primary = MockBackend(
        f"{name}-primary",
        base_latency_ms=primary_base_ms,
        item_latency_ms=primary_item_ms,
        fail_on_marker=True,
    )
    fallback = MockBackend(
        f"{name}-fallback",
        base_latency_ms=12,
        item_latency_ms=2,
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
        drain_timeout_ms=1200,
    )
    await batcher.start()
    semaphore = asyncio.Semaphore(concurrency)
    statuses: Counter[str] = Counter()
    success_latencies: list[float] = []
    batch_sizes: list[int] = []

    async def one(index: int) -> None:
        async with semaphore:
            started = time.perf_counter()
            try:
                outcome = await batcher.submit(
                    f"{name}-{index}",
                    timeout_ms=policy["timeout_ms"],
                )
                statuses["ok"] += 1
                success_latencies.append((time.perf_counter() - started) * 1000)
                batch_sizes.append(outcome.batch_size)
            except QueueFullError:
                statuses["overloaded"] += 1
            except TimeoutError:
                statuses["timeout"] += 1
            except BackendUnavailableError:
                statuses["unavailable"] += 1

    started = time.perf_counter()
    try:
        await asyncio.gather(*(one(index) for index in range(requests)))
    finally:
        await batcher.stop()
    duration = time.perf_counter() - started
    ordered = sorted(success_latencies)
    p95_ms = 0.0
    if ordered:
        p95_index = min(len(ordered) - 1, round((len(ordered) - 1) * 0.95))
        p95_ms = round(ordered[p95_index], 2)
    successful = statuses["ok"]

    return {
        "status_counts": dict(statuses),
        "success_rate": round(successful / requests, 4),
        "p95_ms": p95_ms,
        "admitted_rps": round(successful / duration, 2),
        "shed_requests": statuses["overloaded"],
        "fallback_requests": backend.fallback_requests,
        "primary_timeouts": backend.primary_timeouts,
        "breaker_state": breaker.state.value,
        "mean_batch_size": round(statistics.mean(batch_sizes), 2)
        if batch_sizes
        else 0,
        "primary_backend_calls": primary.batch_calls,
        "primary_items_seen": primary.items_processed,
    }


async def _deadline_waste_scenario() -> dict[str, Any]:
    primary = MockBackend(
        "deadline-primary",
        base_latency_ms=130,
        item_latency_ms=1,
    )
    fallback = MockBackend(
        "deadline-fallback",
        base_latency_ms=10,
        item_latency_ms=1,
    )
    breaker = CircuitBreaker(failure_threshold=3, recovery_seconds=0.08)
    backend = ResilientBackend(
        primary,
        fallback,
        breaker,
        primary_timeout_ms=400,
        fallback_timeout_ms=120,
    )
    batcher = DynamicBatcher(
        backend,
        max_batch_size=1,
        max_wait_ms=1,
        capacity=2,
        drain_timeout_ms=1000,
    )
    await batcher.start()
    first = asyncio.create_task(batcher.submit("occupy-worker", timeout_ms=500))
    await asyncio.sleep(0.01)
    timed_out = False
    try:
        await batcher.submit("expired-before-service", timeout_ms=20)
    except TimeoutError:
        timed_out = True
    await first
    await asyncio.sleep(0.16)
    expired = batcher.expired_requests
    primary_items = primary.items_processed
    await batcher.stop()

    return {
        "passed": timed_out and expired >= 1 and primary_items == 1,
        "expected_control": "expired request is dropped before a model call",
        "client_timed_out": timed_out,
        "expired_before_service": expired,
        "primary_items_seen": primary_items,
        "wasted_model_calls": max(0, primary_items - 1),
    }


async def _failure_recovery_scenario(
    policy: dict[str, int],
) -> dict[str, Any]:
    primary = MockBackend(
        "failure-primary",
        base_latency_ms=24,
        item_latency_ms=3,
        fail_on_marker=True,
    )
    fallback = MockBackend(
        "failure-fallback",
        base_latency_ms=10,
        item_latency_ms=1,
    )
    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_seconds=0.08,
    )
    backend = ResilientBackend(
        primary,
        fallback,
        breaker,
        primary_timeout_ms=policy["primary_timeout_ms"],
        fallback_timeout_ms=120,
    )

    for index in range(3):
        result = await backend.infer_batch([f"__FAIL_PRIMARY__-{index}"])
        assert result[0].fallback_used
    open_state = breaker.state.value
    primary_calls_before_open_request = primary.batch_calls
    while_open = await backend.infer_batch(["healthy-while-open"])
    primary_calls_after_open_request = primary.batch_calls
    await asyncio.sleep(0.09)
    recovered = await backend.infer_batch(["recovery-probe"])
    recovered_state = breaker.state.value
    await backend.aclose()

    pass_condition = (
        open_state == "open"
        and while_open[0].fallback_used
        and primary_calls_after_open_request == primary_calls_before_open_request
        and not recovered[0].fallback_used
        and recovered_state == "closed"
    )
    return {
        "passed": pass_condition,
        "expected_control": (
            "hard failure opens the circuit, skips primary while open, then "
            "restores primary after a successful recovery probe"
        ),
        "open_state": open_state,
        "healthy_while_open_backend": while_open[0].backend,
        "primary_calls_during_open": (
            primary_calls_after_open_request - primary_calls_before_open_request
        ),
        "recovery_backend": recovered[0].backend,
        "recovered_state": recovered_state,
        "fallback_requests": backend.fallback_requests,
    }
