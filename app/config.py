from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "InferenceRail")
    app_version: str = os.getenv("APP_VERSION", "0.4.0")
    revision: str = os.getenv("GIT_SHA") or os.getenv("RENDER_GIT_COMMIT", "local")
    environment: str = os.getenv("APP_ENV", "local")
    backend_mode: str = os.getenv("BACKEND_MODE", "mock").lower()
    queue_capacity: int = _env_int("QUEUE_CAPACITY", 256)
    max_batch_size: int = _env_int("MAX_BATCH_SIZE", 8)
    max_batch_wait_ms: int = _env_int("MAX_BATCH_WAIT_MS", 12)
    request_timeout_ms: int = _env_int("REQUEST_TIMEOUT_MS", 1200)
    shutdown_drain_timeout_ms: int = _env_int("SHUTDOWN_DRAIN_TIMEOUT_MS", 5000)
    primary_timeout_ms: int = _env_int("PRIMARY_TIMEOUT_MS", 700)
    fallback_timeout_ms: int = _env_int("FALLBACK_TIMEOUT_MS", 350)
    breaker_failure_threshold: int = _env_int("BREAKER_FAILURE_THRESHOLD", 3)
    breaker_recovery_seconds: float = _env_float("BREAKER_RECOVERY_SECONDS", 5.0)
    mock_base_latency_ms: float = _env_float("MOCK_BASE_LATENCY_MS", 35.0)
    mock_item_latency_ms: float = _env_float("MOCK_ITEM_LATENCY_MS", 6.0)
    triton_url: str = os.getenv("TRITON_URL", "http://triton:8000")
    triton_model: str = os.getenv("TRITON_MODEL", "text_classifier")
    triton_input_name: str = os.getenv("TRITON_INPUT_NAME", "TEXT")
    triton_label_output: str = os.getenv("TRITON_LABEL_OUTPUT", "LABEL")
    triton_score_output: str = os.getenv("TRITON_SCORE_OUTPUT", "SCORE")
    otlp_endpoint: str | None = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    slo_p95_ms: float = _env_float("SLO_P95_MS", 250.0)
    slo_success_rate: float = _env_float("SLO_SUCCESS_RATE", 0.995)


settings = Settings()
