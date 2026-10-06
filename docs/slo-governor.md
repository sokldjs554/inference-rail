# SLO Governor

InferenceRail v0.5의 대표 기능입니다.

## 문제

Dynamic batching, queue size, timeout, breaker threshold는 각각 개별적으로는 단순한 설정이지만 서로 trade-off를 만듭니다.

- 작은 batch / 짧은 queue: 낮은 latency, 낮은 admission capacity
- 큰 batch / 긴 queue: 높은 throughput, 높은 tail-latency 위험
- 빠른 primary timeout: 장애 격리는 빠르지만 정상적인 느린 요청도 fallback으로 보낼 수 있음

정적 설정을 “best practice”로 고정하지 않고, **같은 workload를 실제 serving path에 replay해 선택 근거를 남기는 것**이 SLO Governor의 목적입니다.

## Counterfactual replay

요청 profile은 세 가지입니다.

- steady
- flash_crowd
- degraded_primary

각 profile은 요청 수, concurrency, primary failure schedule이 고정됩니다. 해당 workload는 세 개의 격리된 serving runtime에서 동일하게 재생됩니다.

후보 정책:

- latency_guard
- throughput_guard
- availability_guard

각 runtime은 실제 프로젝트의 다음 코드를 그대로 사용합니다.

```text
DynamicBatcher
  -> ResilientBackend
      -> CircuitBreaker
      -> Primary MockBackend
      -> Fallback MockBackend
```

mock은 실제 GPU 성능을 주장하기 위한 것이 아니라 gateway policy 차이를 deterministic하게 비교하기 위한 workload입니다.

## Decision rule

사용자가 지정하는 objective:

- target p95
- minimum success rate

각 정책에서 수집:

- success rate
- successful-request p95
- admitted throughput
- shed count
- fallback requests
- backend batch calls
- backend calls / 100 successful requests

SLO를 만족한 정책 중 `backend calls / 100 successful requests`가 가장 낮은 정책을 선택합니다. 이 값은 달러 비용을 임의로 추정하지 않고 batching이 실제 model invocation 수를 얼마나 줄였는지를 보여주는 compute pressure proxy입니다.

## Decision Receipt

응답에는 workload fingerprint와 함께 다음이 기록됩니다.

- selected policy
- selected config
- SLO pass/fail
- 각 대안의 측정 결과
- 탈락 이유
- 선택 규칙

API:

```http
POST /v1/slo-decision
Content-Type: application/json

{
  "profile": "flash_crowd",
  "target_p95_ms": 250,
  "min_success_rate": 0.995
}
```

공개 데모에서는 shared serving runtime을 변경하지 않습니다. 격리 replay로 추천 config를 발행하는 방식이라 여러 사용자가 동시에 데모를 보더라도 서로의 runtime 설정을 바꾸지 않습니다.
