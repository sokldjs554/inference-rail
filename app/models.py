from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class PredictResponse(BaseModel):
    request_id: str
    label: str
    score: float
    backend: str
    fallback_used: bool
    batch_size: int
    queue_ms: float
    service_ms: float


class PolicyLabRequest(BaseModel):
    profile: Literal["steady", "flash_crowd", "degraded_primary"] = "flash_crowd"


class SLODecisionRequest(PolicyLabRequest):
    target_p95_ms: float = Field(default=250.0, ge=50.0, le=2000.0)
    min_success_rate: float = Field(default=0.995, ge=0.5, le=1.0)


class ServingConfig(BaseModel):
    queue_capacity: int = Field(ge=1, le=4096)
    batch_size: int = Field(ge=1, le=128)
    batch_wait_ms: int = Field(ge=0, le=1000)
    timeout_ms: int = Field(ge=50, le=10000)
    primary_timeout_ms: int = Field(ge=20, le=10000)
    breaker_threshold: int = Field(ge=1, le=20)


class ServiceBoundaryRequest(SLODecisionRequest):
    config: ServingConfig | None = None


class RuntimeStatus(BaseModel):
    backend_mode: str
    queue_depth: int
    queue_capacity: int
    breaker_state: str
    breaker_failures: int
    processed_requests: int
    expired_requests: int
    fallback_requests: int
    primary_failures: int
    primary_timeouts: int
    fallback_failures: int


class BuildInfo(BaseModel):
    app: str
    version: str
    revision: str
    environment: str
    backend_mode: str
