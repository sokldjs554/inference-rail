# InferenceRail

**트래픽 급증과 모델 장애가 발생해도 서비스 경계를 지키는 ML inference gateway를 검증한 프로젝트입니다.**

토스뱅크 ML Backend Engineer 지원을 목표로, 단순한 FastAPI 예제가 아니라 **대규모 요청을 받는 모델 서빙 서버에서 실제로 문제가 되는 queue saturation, tail latency, backend timeout, 장애 전파, fallback, observability, autoscaling**을 한 저장소에서 재현하도록 만들었습니다.

> 기본 backend는 deterministic mock입니다. 모델 정확도를 꾸미는 프로젝트가 아니라 서버 구조의 특성을 반복 측정하기 위한 선택입니다. 실제 모델 서버 경계는 `BACKEND_MODE=triton`으로 분리했고, NVIDIA Triton V2 HTTP request/response contract를 사용하는 adapter와 별도 프로세스 기반 contract 검증을 포함했습니다.

## 바로 확인하기

- **공개 데모:** https://inference-rail-demo.onrender.com
- **GitHub Actions 검증:** https://github.com/sokldjs554/inference-rail/actions/runs/37438191739
- **현재 공개 배포 revision:** `a37d74082ba26af1f9a5e596d68d2acf5fffcdcd`

공개 데모는 Render Singapore에서 `BACKEND_MODE=mock`으로 운영합니다. 배포 후 외부 브라우저에서 `/`, `/health/ready`, `/ops/build`, `/ops/status`를 확인했고, 데모의 `정상 요청` 버튼을 실제 클릭해 **HTTP 200 / primary-model / fallback=false** 응답까지 검증했습니다.

## 한눈에 보기

```text
Client request + deadline
  ↓
FastAPI Gateway
  ↓
Bounded Queue ───────────────→ queue full: 429 fast rejection
  ↓ expired before service   → drop before model call
Dynamic Batcher
  ↓
Primary Backend ── timeout/failure ─→ Circuit Breaker
  ↓                                      ↓
Response ←──────────── Fallback Backend ←─┘
  ↓
Prometheus + OpenTelemetry span links
  ↓
CPU HPA or queue-saturation KEDA scaling
```

핵심은 “FastAPI를 사용했다”가 아닙니다. **어떤 실패 모드를 가정했고, 왜 그 대응을 선택했고, 그 선택을 어떤 지표로 검증했는지**를 설명할 수 있도록 구성했습니다.

## 현재 검증 결과

`PYTHONPATH=. python scripts/release_verify.py`로 테스트, 운영 시나리오, Triton contract, manifest, benchmark, SLO gate를 한 번에 다시 검증할 수 있습니다.

최근 release 검증 결과:

| 검증 | 결과 |
|---|---:|
| Python tests | **10 / 10 pass** |
| Triton V2 contract E2E | **8 requests / batch size 8 / fallback 0** |
| Circuit breaker | **3회 primary 실패 → open → fallback → recovery 후 closed** |
| 강제 overload | **60건 중 57건 429, 3건 정상 admission** |
| Batching throughput | **24.09~24.10 → 125.33~125.60 req/s, 5.20~5.21×** |
| p95 latency | **996.30~996.88 → 189.06~190.43 ms, -80.89~-81.02%** |
| Batched success rate | **100%** |
| YAML/운영 설정 | **15개 파일 parse/contract 검증 통과** |
| SLO gate | **pass** |
| GitHub Actions | **lint + release verification + Docker build + container smoke pass** |
| Render 공개 배포 | **live / revision 일치 / browser predict pass** |

벤치마크 조건은 `160 requests / concurrency 24`, deterministic mock workload입니다. **실제 GPU 또는 실제 모델의 성능 수치가 아니라 gateway batching 구조의 차이를 분리해 측정한 값**입니다.

원본 검증 증거:

- `docs/evidence/release-verification.json`
- `docs/evidence/release-benchmark.json`
- `docs/evidence/release-verification-run-1.json` ~ `release-verification-run-3.json`
- `docs/evidence/release-benchmark-run-1.json` ~ `release-benchmark-run-3.json`
- `docs/verification-20261006.md`
- `docs/evidence/public-render-smoke.json`

## 왜 이 구조인가

### 1. 무제한 queue 대신 bounded queue

모델 처리량보다 입력률이 높아질 때 무제한 queue는 순간적인 오류율은 낮출 수 있지만 요청이 계속 쌓이면서 tail latency와 메모리 사용량을 통제하기 어려워집니다.

InferenceRail은 queue capacity를 넘으면 **429로 빠르게 거절**합니다. release 검증에서는 capacity를 2로 강제로 낮추고 60건을 동시에 보내 **57건을 빠르게 거절하고 3건만 admission**하는 동작을 확인했습니다.

### 2. client timeout만 두지 않고 deadline-aware queue 사용

HTTP 요청이 timeout된 뒤에도 queue의 작업이 모델로 전달되면 이미 응답할 수 없는 요청이 GPU/CPU 시간을 소비합니다.

그래서 각 queue item에 deadline을 저장하고:

1. 기다리는 동안 client deadline이 지나면 future를 취소하고,
2. worker가 dequeue할 때 expired item을 제거하고,
3. 모델 호출 전 불필요한 작업을 버립니다.

### 3. scale-out 전에 dynamic batching

모델 inference는 호출마다 고정 비용이 존재합니다. 트래픽 증가를 pod 증설로만 해결하면 비용이 빠르게 증가할 수 있습니다.

짧은 batch window를 허용해 여러 요청을 한 inference call로 묶는 선택을 검증했습니다.

```text
MAX_BATCH_SIZE=1   → 24.09~24.10 req/s / p95 996.30~996.88 ms
MAX_BATCH_SIZE=8   → 125.33~125.60 req/s / p95 189.06~190.43 ms
```

최종 3회 반복에서 throughput 개선은 **5.20~5.21×**, 평균 **5.20×**였습니다.

### 4. backend별 timeout + circuit breaker + fallback

무거운 inference의 무조건 retry는 이미 느려지거나 장애 난 backend에 추가 요청을 보내 retry storm을 만들 수 있습니다.

InferenceRail은:

- primary timeout budget
- fallback timeout budget
- failure threshold
- circuit open/recovery window

을 분리했습니다.

검증에서는 primary를 3회 강제 실패시켜 breaker를 `open`으로 만든 뒤, 정상 요청도 primary를 건드리지 않고 fallback으로 처리하는지 확인했습니다. recovery window 이후 probe가 성공하면 다시 `closed`로 복귀합니다.

### 5. batching 때문에 끊기기 쉬운 trace를 span link로 연결

HTTP request span과 batch worker는 비동기 경계가 다르고, 여러 request가 하나의 model batch로 합쳐집니다. 단순 parent-child trace로는 이 관계를 정확히 표현하기 어렵습니다.

각 request의 span context를 queue item에 보존한 뒤 `inference.batch` span에 **OpenTelemetry Link**로 연결합니다. batch 내부에서는 `inference.primary` / `inference.fallback` span을 남깁니다.

즉 “요청 하나가 어느 모델 호출에 들어갔는가”뿐 아니라 **여러 요청이 하나의 batch로 합쳐진 관계**까지 추적할 수 있는 구조입니다.

## 실제 모델 서버 경계: Triton V2

`BACKEND_MODE=triton`이면 primary를 `TritonHTTPBackend`로 교체합니다.

기본 contract:

```text
POST /v2/models/{model}/infer

Input
  TEXT: BYTES [batch, 1]

Output
  LABEL: one label per item
  SCORE: one float per item
```

설정:

```bash
BACKEND_MODE=triton
TRITON_URL=http://triton:8000
TRITON_MODEL=text_classifier
TRITON_INPUT_NAME=TEXT
TRITON_LABEL_OUTPUT=LABEL
TRITON_SCORE_OUTPUT=SCORE
```

release verification은 별도 프로세스로 Triton V2 contract stub을 실행하고, gateway도 별도 프로세스로 `BACKEND_MODE=triton`으로 띄웁니다. 그 상태에서 8개의 동시 요청을 보내 **실제 HTTP network boundary를 왕복해 batch size 8, `triton:text_classifier`, fallback 0건**을 확인합니다.

> 이 검증은 NVIDIA Triton HTTP contract adapter의 동작을 확인하는 테스트입니다. 현재 실행 환경에서는 실제 NVIDIA Triton Server/GPU를 띄우지 않았으므로 실제 Triton runtime 성능 검증이라고 주장하지 않습니다.

## 빠른 실행

Python 3.11+ 권장:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
uvicorn app.main:app --reload
```

브라우저에서 `http://127.0.0.1:8000`을 열면 다음을 직접 재현할 수 있습니다.

- 정상 요청
- 24개 동시 요청과 batch size 확인
- primary 장애 1회
- 3회 장애로 circuit breaker open
- backend mode / queue / breaker / fallback / timeout 상태 확인

API 예시:

```bash
curl -s http://127.0.0.1:8000/v1/predict \
  -H 'content-type: application/json' \
  -d '{"text":"hello"}'
```

강제 primary failure:

```bash
curl -s http://127.0.0.1:8000/v1/predict \
  -H 'content-type: application/json' \
  -d '{"text":"__FAIL_PRIMARY__"}'
```

## 재현 가능한 전체 검증

```bash
PYTHONPATH=. python scripts/release_verify.py
```

이 명령은 다음을 순서대로 검증합니다.

1. Python tests
2. Docker Compose / Prometheus / OTel / Grafana / Kubernetes YAML contract
3. Triton V2 HTTP contract를 통한 remote backend mode
4. circuit breaker open → fallback → recovery
5. bounded queue overload와 429 fast rejection
6. batching on/off benchmark
7. p95 + success-rate SLO gate

개별 실행:

```bash
python -m pytest -q
PYTHONPATH=. python scripts/validate_manifests.py
PYTHONPATH=. python scripts/benchmark_compare.py --requests 160 --concurrency 24
python scripts/slo_gate.py benchmark-results.json
```


## 배포와 provenance

공개 데모는 `render.yaml`과 동일한 Python runtime 설정으로 **실제 Render에 배포했습니다**. 공개 데모에서는 재현성을 위해 deterministic mock backend를 사용하고, 실제 모델 서버 경계는 별도의 Triton adapter/contract 검증으로 분리합니다.

- 공개 URL: https://inference-rail-demo.onrender.com
- 배포 region: Singapore
- 배포 revision: `a37d74082ba26af1f9a5e596d68d2acf5fffcdcd`
- 외부 검증: `/health/ready`, `/ops/build`, `/ops/status`, UI `정상 요청` 모두 성공

GitHub Actions에서도 같은 revision을 대상으로 **release verification → Docker image build → 실제 container start → health/build/predict smoke**까지 통과했습니다.

배포 뒤에는 다음 endpoint로 **실행 중인 코드가 어떤 revision인지** 확인할 수 있습니다.

```text
GET /ops/build
```

예시:

```json
{
  "app": "InferenceRail",
  "version": "0.3.0",
  "revision": "<deployment git sha>",
  "environment": "<runtime>",
  "backend_mode": "mock"
}
```

`scripts/public_smoke.py <base-url>`은 health, build provenance, predict, runtime status를 한 번에 확인합니다. 배포 구조와 검증 경계는 `docs/deployment.md`, 주요 트레이드오프는 `docs/design-decisions.md`에 기록했습니다.

## Observability

Docker 환경에서는 다음 구성을 사용합니다.

```bash
docker compose up --build
```

- Application / demo: `:8000`
- Prometheus: `:9090`
- Grafana: `:3000`
- Jaeger: `:16686`

주요 metric:

- `inference_requests_total`
- `inference_request_latency_seconds`
- `inference_queue_latency_seconds`
- `inference_batch_size`
- `inference_queue_depth`
- `inference_queue_capacity`
- `inference_fallback_total`
- `inference_deadline_expired_total`
- `inference_backend_errors_total`
- `inference_circuit_breaker_state`

Prometheus alert rule도 포함합니다.

- p95 > 250 ms
- queue saturation > 70%
- fallback 비율 > 20%
- backend unavailable 응답 발생

Grafana에는 p95, throughput/status, queue depth/capacity, fallback traffic 패널을 provision합니다.

## Kubernetes 운영 contract

기본 배포:

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/pdb.yaml
kubectl apply -f k8s/hpa.yaml
```

포함 항목:

- startup / readiness / liveness probe
- `maxUnavailable: 0` rolling update
- graceful termination + preStop
- PodDisruptionBudget
- non-root user
- privilege escalation 차단
- read-only root filesystem
- Linux capability drop
- topology spread
- resource requests/limits

### CPU HPA와 queue 기반 KEDA

`k8s/hpa.yaml`은 어디서나 이해하기 쉬운 CPU 기반 fallback 예시입니다.

모델 서빙은 CPU보다 pending request/queue saturation이 실제 병목을 더 잘 나타낼 수 있으므로 `k8s/keda-scaledobject.yaml`에는 Prometheus query 기반 scaling도 별도로 넣었습니다.

```text
sum(inference_queue_depth)
────────────────────────────── > 0.60
sum(inference_queue_capacity)
```

> `hpa.yaml`과 `keda-scaledobject.yaml`을 같은 Deployment에 동시에 적용하지 않습니다. KEDA 예시는 cluster에 KEDA와 Prometheus가 설치되어 있다는 전제입니다.

`k8s/servicemonitor.yaml`은 Prometheus Operator 환경용 optional contract입니다.

## 디렉터리 구조

```text
inference-rail/
├── app/
│   ├── backends/
│   │   ├── mock.py
│   │   ├── resilient.py
│   │   └── triton.py
│   ├── core/
│   │   ├── batcher.py
│   │   └── circuit_breaker.py
│   ├── static/index.html
│   ├── main.py
│   ├── metrics.py
│   └── telemetry.py
├── tests/
├── scripts/
│   ├── benchmark_compare.py
│   ├── fault_scenario.py
│   ├── load_test.py
│   ├── release_verify.py
│   ├── slo_gate.py
│   ├── triton_contract_stub.py
│   ├── public_smoke.py
│   └── validate_manifests.py
├── docs/
│   ├── design-decisions.md
│   ├── deployment.md
│   └── evidence/
├── k8s/
├── prometheus/
├── grafana/
├── otel/
├── Dockerfile
├── docker-compose.yml
└── render.yaml
```

## 포트폴리오에서 설명할 핵심

면접에서는 기술 이름 나열보다 아래 순서로 설명하는 것을 목표로 합니다.

**문제 → 선택지 → 트레이드오프 → 구현 → 실제 측정 → 남은 한계**

예를 들면:

> 모델 요청이 처리량을 넘어설 때 무제한 queue를 유지하면 tail latency가 지속적으로 커지고 이미 timeout된 요청까지 모델 자원을 소비한다고 판단했습니다. queue capacity와 request deadline을 분리해 admission control을 만들고, dequeue 시 deadline이 지난 작업을 모델 호출 전에 제거했습니다. 동시에 짧은 dynamic batching window를 적용했고, 고정 workload의 최종 3회 반복에서 batch size 1은 24.09~24.10 req/s, batch size 8은 125.33~125.60 req/s였습니다. 과부하 조건에서는 60건 중 57건을 429로 빠르게 차단해 시스템이 처리 가능한 요청만 admission하는 동작을 검증했습니다.

이런 식으로 **서버 선택의 이유와 결과 지표를 함께 설명하는 것**이 이 프로젝트의 목적입니다.

## 현재 한계 — 숨기지 않는 부분

현재 환경에서 실제로 검증한 것과 아직 하지 않은 것을 구분합니다.

검증 완료:

- Python 기능/회귀 테스트
- deadline-aware queue
- primary/fallback timeout
- breaker failure/recovery
- overload fast rejection
- dynamic batching benchmark
- Triton V2 HTTP adapter의 별도 프로세스 network contract
- YAML 구조 및 주요 운영 contract
- Prometheus/Grafana/OTel 설정 구조
- Render 배포 manifest contract
- `/ops/build` deployment provenance + 공개 URL smoke script
- GitHub Actions Linux runner에서 실제 Docker image build
- 빌드한 Docker container 실제 기동 후 health/build/predict smoke
- Render 공개 배포 및 외부 브라우저 정상 예측

아직 실제 runtime 검증하지 못한 것:

- Docker Compose의 Prometheus/Grafana/Jaeger 전체 multi-container stack 기동
- 실제 Kubernetes cluster에서 rollout / HPA / KEDA scale-out
- 실제 NVIDIA Triton Server + GPU inference
- 실제 GPU utilization / VRAM / model-level throughput

따라서 이 저장소의 현재 수치를 실제 은행 production SLA나 GPU 성능으로 주장하지 않습니다.

## License

`All rights reserved.` 형태이며 MIT license를 사용하지 않습니다.
