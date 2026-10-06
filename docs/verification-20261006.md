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

## 추가 runtime 검증으로 해소한 항목

초기 실행 환경에서 직접 확인하지 못했던 Docker/Kubernetes/Triton 항목은 GitHub Actions Linux runner에서 별도 runtime workflow로 다시 검증했습니다.

현재까지 실제 runtime으로 확인한 항목:

- Docker image build / 실제 container smoke
- Docker Compose 전체 observability stack
- kind Kubernetes 1.37 rollout
- Metrics Server 기반 CPU HPA scale-out
- Prometheus/KEDA 기반 queue saturation scale-out
- NVIDIA Triton Inference Server 2.73.0 실제 container + CPU Python backend

현재 남은 미검증 범위는 **실제 GPU-backed model inference와 GPU utilization/VRAM/throughput**입니다.


## GitHub Actions / Docker runtime 검증

커밋 `a37d74082ba26af1f9a5e596d68d2acf5fffcdcd`에서 GitHub Actions run #2를 실행했습니다.

- Ruff lint: **pass**
- release verification: **pass**
- Docker image build: **pass**
- 실제 container start: **pass**
- container `/health/ready`: **pass**
- container `/ops/build`: **pass**
- container `/v1/predict`: **pass**
- release evidence artifact upload: **pass**

Actions: https://github.com/sokldjs554/inference-rail/actions/runs/37438191739

## Render 공개 배포 검증

공개 URL: https://inference-rail-demo.onrender.com

배포 revision: `a37d74082ba26af1f9a5e596d68d2acf5fffcdcd`

외부 검증 결과:

- `/`: InferenceRail demo page 정상 로드
- `/health/ready`: `status=ready, queue_depth=0`
- `/ops/build`: version `0.3.0`, environment `render`, revision 일치
- `/ops/status`: backend `mock`, queue `0/256`, breaker `closed`
- 브라우저에서 `정상 요청` 버튼 1회 실행: HTTP 200, `primary-model`, `fallback_used=false`, batch size 1

원본 요약은 `docs/evidence/public-render-smoke.json`에 보존했습니다.


## Kubernetes 실제 runtime — 2026-10-06

성공 run: https://github.com/sokldjs554/inference-rail/actions/runs/37451336671

- kind: **v0.33.0**
- Kubernetes: **1.37**
- application rollout + Service smoke: **pass**
- Metrics Server: **v0.9.0**
- HPA 관측 CPU: **61% / CI target 5%**
- HPA scale: **2 → 6 desired replicas**
- Prometheus + KEDA: **2.21.0**
- queue saturation external metric: **500m**
- CI KEDA target: **400m (0.40)**
- KEDA scale: **2 → 3 desired replicas**
- production HPA target: **65% 유지**
- production KEDA threshold: **0.60 유지**

CI의 낮은 임계값은 실제 control path를 안정적으로 재현하기 위한 live-object patch이며 운영 manifest 값은 변경하지 않았습니다.

영구 요약: `docs/evidence/k8s-runtime-summary.json`

## 실제 NVIDIA Triton runtime — 2026-10-06

성공 run: https://github.com/sokldjs554/inference-rail/actions/runs/37450818648

- NVIDIA container: **26.09-py3**
- Triton Inference Server: **2.73.0**
- model: `text_classifier` Python backend / **KIND_CPU**
- direct Triton V2 batch: **4 / HTTP 200**
- gateway concurrent requests: **8**
- gateway observed batch size: **8**
- backend: **triton:text_classifier**
- fallback: **false**
- primary failures/timeouts: **0 / 0**
- breaker: **closed**

이 결과는 stub이 아닌 실제 Triton Server runtime입니다. 단, CPU instance이므로 GPU 성능 결과는 아닙니다.

영구 요약: `docs/evidence/triton-runtime-summary.json`

## Docker Compose observability runtime — 2026-10-06

성공 run: https://github.com/sokldjs554/inference-rail/actions/runs/37452240152

실제 기동한 구성:

- InferenceRail app
- Prometheus 2.55.1
- Grafana 11.2.2
- Jaeger 1.62.0
- OpenTelemetry Collector 0.111.0

예측 요청 이후:

- app health/predict: **pass**
- Prometheus `inference_requests_total` query: **non-empty**
- Grafana database health: **ok**
- Jaeger services: **inference-rail 확인**
- Jaeger trace query: **traceID 확인**

영구 요약: `docs/evidence/compose-runtime-summary.json`
