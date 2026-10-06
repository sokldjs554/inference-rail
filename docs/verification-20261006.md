# InferenceRail 검증 기록 — 2026-10-06

## 목적

토스뱅크 ML Backend Engineer 지원용 프로젝트가 코드 존재만이 아니라 실제 실행 결과로 다음 특성을 증명하는지 확인했습니다.

- bounded queue / overload rejection
- dynamic batching
- request deadline
- primary/fallback timeout
- circuit breaker + recovery
- Triton V2 remote backend contract
- observability configuration
- Kubernetes deployment contract
- SLO gate

## 최종 release verification

실행 명령:

```bash
PYTHONPATH=. python scripts/release_verify.py
```

### 1. Python tests

결과: **10 passed**

검증 범위:

- 정상 inference API
- primary failure → fallback
- dynamic batching
- bounded queue overflow
- graceful drain
- queue에서 deadline 만료된 요청을 backend로 보내지 않음
- circuit breaker open
- half-open recovery
- primary backend timeout → fallback
- Triton V2 request/response parsing

### 2. 운영/manifest contract

YAML parse 및 필수 contract 검증: **pass**

검증 파일 14개:

- Docker Compose
- Prometheus scrape/alert rule
- OpenTelemetry Collector
- Grafana datasource/dashboard provisioning
- Kubernetes ConfigMap / Triton ConfigMap / Deployment / Service / HPA / PDB
- KEDA ScaledObject
- ServiceMonitor

Deployment에 startup/readiness/liveness probe와 privilege escalation 차단이 존재하는지 추가 확인했습니다.

### 3. Triton remote backend contract

별도 프로세스로 Triton V2 HTTP contract stub을 실행한 뒤 gateway를 `BACKEND_MODE=triton`으로 실행했습니다.

결과:

- 동시 요청: **8**
- gateway batch size: **8**
- backend: **`triton:text_classifier`**
- fallback: **0**
- primary failure: **0**
- primary timeout: **0**
- breaker: **closed**

이는 실제 HTTP network boundary와 Triton V2 payload adapter를 검증한 결과입니다. 실제 NVIDIA Triton runtime/GPU 성능 검증은 아닙니다.

### 4. Circuit breaker fault scenario

설정:

- failure threshold: 3
- 검증용 recovery window: 0.15 s

실행 순서:

1. primary 강제 실패 1회 → fallback
2. primary 강제 실패 2회 → fallback
3. primary 강제 실패 3회 → breaker `open`
4. 정상 요청 → primary를 호출하지 않고 fallback
5. recovery window 경과
6. 정상 probe → primary 성공 → breaker `closed`

결과: **pass**

### 5. Overload / backpressure

강제 조건:

- queue capacity: 2
- max batch size: 1
- mock base latency: 120 ms
- requests: 60
- concurrency: 40

결과:

- HTTP 429: **57**
- HTTP 200: **3**
- 전체 시나리오 시간: **약 0.39 s**

queue에 무제한 적재하지 않고 처리 가능한 요청만 admission하고 나머지는 빠르게 거절하는 동작을 확인했습니다.

### 6. Dynamic batching benchmark

조건:

- requests: 160
- concurrency: 24
- 같은 deterministic mock workload
- 최종 코드 상태로 3회 반복

| 지표 | Batch 1 (3회 범위) | Batch 8 (3회 범위) |
|---|---:|---:|
| Throughput | 24.09~24.10 req/s | **125.33~125.60 req/s** |
| p95 | 996.30~996.88 ms | **189.06~190.43 ms** |
| Success rate | 3회 모두 100% | **3회 모두 100%** |
| Mean batch size | 1 | **8** |

- throughput gain: **5.20~5.21×**
- p95 change: **-80.89~-81.02%**
- batch 8 평균 throughput: **125.44 req/s**
- batch 8 평균 p95: **189.81 ms**

각 반복에서 `SLO p95 <= 250 ms`, `success rate >= 99.5%` gate: **3회 모두 pass**

최종 tracing/dependency/Dockerfile 수정 후 전체 release smoke도 다시 실행했고 **10 tests, Triton contract, breaker recovery, overload, manifest, SLO gate 모두 pass**했습니다. 최종 smoke 원본은 `release-verification-final.json` / `release-benchmark-final.json`에 보존했습니다.

> 위 수치는 모델 정확도나 GPU 성능이 아니라 gateway batching 구조의 효과를 재현 가능한 mock workload에서 측정한 결과입니다.

## 증거 파일

- `docs/evidence/release-verification.json`
- `docs/evidence/release-benchmark.json`
- `docs/evidence/release-verification-run-1.json` ~ `release-verification-run-3.json`
- `docs/evidence/release-benchmark-run-1.json` ~ `release-benchmark-run-3.json`
- `docs/evidence/release-verification-final.json` / `release-benchmark-final.json`
- 기존 초기 반복 benchmark: `benchmark-run-1.json` ~ `benchmark-run-3.json`

## 현재 실행 환경에서 검증하지 못한 부분

이 환경에는 Docker daemon / kubectl / GPU runtime이 없었습니다. 따라서 다음은 코드/manifest와 정적 contract만 준비했고 실제 runtime 성공으로 보고하지 않습니다.

- `docker build`
- `docker compose up` 전체 stack
- Kubernetes rollout
- CPU HPA 실제 scale-out
- KEDA queue metric 실제 scale-out
- 실제 NVIDIA Triton Server
- 실제 GPU utilization/VRAM benchmark

실제 cluster/GPU 환경에서 이 항목을 수행하기 전까지는 production-grade runtime 완료라고 표현하지 않습니다.
