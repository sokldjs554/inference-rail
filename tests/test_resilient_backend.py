import pytest

from app.backends.mock import MockBackend
from app.backends.resilient import ResilientBackend
from app.core.circuit_breaker import BreakerState, CircuitBreaker


@pytest.mark.asyncio
async def test_primary_timeout_falls_back_and_counts_timeout():
    primary = MockBackend("slow-primary", base_latency_ms=80, item_latency_ms=1)
    fallback = MockBackend("fast-fallback", base_latency_ms=1, item_latency_ms=1)
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=10)
    backend = ResilientBackend(
        primary,
        fallback,
        breaker,
        primary_timeout_ms=10,
        fallback_timeout_ms=50,
    )

    result = await backend.infer_batch(["one"])

    assert result[0].backend == "fast-fallback"
    assert result[0].fallback_used is True
    assert backend.primary_timeouts == 1
    assert backend.fallback_requests == 1
    assert breaker.state is BreakerState.OPEN
