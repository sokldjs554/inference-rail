from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    pass


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    recovery_seconds: float = 5.0

    def __post_init__(self) -> None:
        self.state = BreakerState.CLOSED
        self.failures = 0
        self.opened_at: float | None = None

    def before_call(self) -> None:
        if self.state is not BreakerState.OPEN:
            return
        assert self.opened_at is not None
        if time.monotonic() - self.opened_at >= self.recovery_seconds:
            self.state = BreakerState.HALF_OPEN
            return
        raise CircuitOpenError("primary backend circuit is open")

    def record_success(self) -> None:
        self.failures = 0
        self.state = BreakerState.CLOSED
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.state is BreakerState.HALF_OPEN or self.failures >= self.failure_threshold:
            self.state = BreakerState.OPEN
            self.opened_at = time.monotonic()
