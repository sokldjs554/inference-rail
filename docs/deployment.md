# 배포 구조와 검증 경계

InferenceRail은 **공개 데모**와 **production-like ML serving 구성**을 의도적으로 분리합니다.

## 공개 데모: Render + deterministic mock backend

공개 URL에서는 GPU 비용이나 외부 모델 서버 가용성 때문에 데모가 흔들리지 않도록 `BACKEND_MODE=mock`을 사용합니다.
데모의 목적은 모델 정확도가 아니라 다음 서버 동작을 재현하는 것입니다.

- dynamic batching
- bounded queue / overload 429
- request deadline
- circuit breaker
- fallback
- runtime status

`render.yaml`은 다음 명령으로 서비스를 실행하도록 정의합니다.

```text
pip install .
uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
```

배포 뒤 확인할 endpoint:

```text
GET  /health/live
GET  /health/ready
GET  /ops/status
GET  /ops/build
POST /v1/predict
GET  /metrics
```

## production-like 구성: Kubernetes + Triton

실제 모델 서버 경계는 `BACKEND_MODE=triton`으로 분리합니다.

```text
Client
  -> InferenceRail Pod(s)
       -> Triton V2 HTTP endpoint
       -> fallback backend
       -> Prometheus metrics
       -> OpenTelemetry collector
```

Kubernetes manifest에는 다음 운영 경계를 포함합니다.

- 2 replicas
- rolling update (`maxUnavailable: 0`)
- startup/readiness/liveness probe
- PDB
- topology spread
- non-root / read-only root filesystem
- graceful termination
- CPU HPA
- optional Prometheus/KEDA queue saturation scaling

## 배포 증거에서 주장하지 않는 것

공개 Render 데모는 실제 NVIDIA Triton/GPU 성능 검증이 아닙니다.
저장소의 Triton 검증은 별도 HTTP 프로세스와 V2 contract를 통과하는 adapter/E2E 검증입니다.
실제 GPU 환경에서 얻지 않은 처리량이나 latency를 GPU 성능으로 표현하지 않습니다.
