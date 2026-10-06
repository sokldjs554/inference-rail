import pytest

from app.core.circuit_breaker import BreakerState, CircuitBreaker, CircuitOpenError


def test_breaker_opens_after_threshold():
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=10)
    breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    with pytest.raises(CircuitOpenError):
        breaker.before_call()


def test_breaker_half_open_can_recover():
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=0)
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    breaker.before_call()
    assert breaker.state is BreakerState.HALF_OPEN
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED
