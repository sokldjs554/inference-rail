# Real NVIDIA Triton runtime verification

빠른 release gate의 contract stub과 별개로 **실제 NVIDIA Triton Inference Server**를 Docker container로 실행해 gateway 경로를 검증했습니다.

## Runtime

- container: `nvcr.io/nvidia/tritonserver:26.09-py3`
- Triton Inference Server: **2.73.0**
- model: `text_classifier`
- backend: Triton Python backend
- instance: **KIND_CPU**
- workflow: https://github.com/sokldjs554/inference-rail/actions/runs/37450818648

## 실제 검증

직접 Triton V2 호출:

- batch: **4**
- HTTP: **200**
- LABEL/SCORE 각 4개 반환

InferenceRail을 통한 호출:

- concurrent requests: **8**
- dynamic batch: **8**
- backend: **triton:text_classifier**
- fallback: **false**
- primary failure: **0**
- primary timeout: **0**
- breaker: **closed**
- build revision: `e20db0989f8f928d38e06e3684e1c3779793b588`

즉 다음 경로를 stub 없이 실제로 통과했습니다.

```text
Client
  → InferenceRail bounded queue / dynamic batcher
  → Triton V2 HTTP
  → NVIDIA Triton Server 2.73.0
  → Python backend model
  → InferenceRail response
```

## 증거

Actions artifact:

- artifact ID: **11405678297**
- digest: `sha256:1a6ba43aae5012325e13d10ca1342286dc48b4648c4a084ed51819193b527413`

artifact에는 Triton server metadata, model metadata, Prometheus-format Triton metrics, server log, runtime summary JSON을 포함합니다.

## 경계

이 검증은 **실제 Triton Server runtime** 검증입니다. 다만 GitHub-hosted runner에 NVIDIA GPU가 없어서 model instance는 CPU입니다. 따라서 GPU latency, VRAM, utilization, power, GPU model throughput은 별도 GPU 환경에서 측정해야 합니다.
