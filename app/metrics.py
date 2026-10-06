from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "inference_requests_total",
    "Inference requests",
    ["status", "backend", "fallback"],
)
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "End-to-end inference request latency",
    buckets=(0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0),
)
QUEUE_LATENCY = Histogram(
    "inference_queue_latency_seconds",
    "Time spent waiting in bounded queue",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
BATCH_SIZE = Histogram(
    "inference_batch_size",
    "Observed dynamic batch size",
    buckets=(1, 2, 4, 8, 16, 32),
)
QUEUE_DEPTH = Gauge("inference_queue_depth", "Current queue depth")
QUEUE_CAPACITY = Gauge("inference_queue_capacity", "Configured bounded queue capacity")
FALLBACK_TOTAL = Counter("inference_fallback_total", "Requests served by fallback backend")
DEADLINE_EXPIRED_TOTAL = Counter(
    "inference_deadline_expired_total",
    "Queued requests dropped because their client deadline expired",
)
BACKEND_ERRORS = Counter(
    "inference_backend_errors_total",
    "Backend errors observed by the gateway",
    ["stage", "reason"],
)

CIRCUIT_STATE = Gauge(
    "inference_circuit_breaker_state",
    "Circuit breaker state as a one-hot gauge",
    ["state"],
)
