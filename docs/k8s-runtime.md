# Kubernetes runtime verification

이 검증은 정적 YAML 검사가 아니라 GitHub Actions Linux runner 안에서 **실제 kind Kubernetes cluster**를 생성해 운영 경계를 확인합니다.

## 검증 범위

1. kind v0.33.0 + Kubernetes 1.37 cluster 생성
2. 현재 커밋으로 Docker image build
3. image를 kind node에 load
4. Deployment / Service / PDB 실제 적용
5. rollout 완료 후 Service port-forward
6. live health / build provenance / predict smoke
7. Metrics Server v0.9.0 설치
8. 실제 HPA가 CPU resource metric을 받아 replicas 2 초과로 scale-out
9. Prometheus + KEDA 2.21.0 설치
10. queue saturation PromQL trigger로 KEDA가 replicas 2 초과로 scale-out

## HPA threshold 주의

공개 데모의 mock backend는 모델 계산을 흉내 내기 위해 대부분의 시간을 async sleep으로 소비합니다. 따라서 CI HPA 검증에서는 **HPA target만 5%로 임시 패치**합니다. 목적은 “5%가 적절한 production threshold”를 주장하는 것이 아니라 Metrics Server → HPA → Deployment의 실제 control-plane 경로가 작동하는지 확인하는 것입니다.

production manifest의 기본값은 65%를 유지합니다.

## KEDA 검증

KEDA 검증에서는 queue saturation 자체가 scale signal이 되도록 임시로 다음 값을 사용합니다.

- queue capacity: 16
- batch size: 1
- mock base latency: 250 ms
- production KEDA threshold: 0.60
- CI live-object KEDA threshold: 0.40

Prometheus는 /metrics를 scrape하고 기존 k8s/keda-scaledobject.yaml의 query를 그대로 평가합니다.

## 증거

workflow artifact k8s-runtime-evidence에 다음을 보존합니다.

- node / pod / deployment 상태
- HPA live/describe 출력
- ScaledObject YAML/describe
- KEDA가 생성한 HPA
- Prometheus queue query 결과
- Metrics Server kubectl top
- KEDA operator log


## 실제 성공 결과

성공 workflow: https://github.com/sokldjs554/inference-rail/actions/runs/37451336671

### HPA

Metrics Server가 실제 resource metric을 제공했고 HPA는 다음 상태를 기록했습니다.

- CPU current / CI target: **61% / 5%**
- replicas: **2 → 6 desired**
- AbleToScale: **True / SucceededRescale**
- ScalingActive: **True / ValidMetricFound**

### KEDA

Prometheus에서 queue saturation을 external metric으로 제공했고 KEDA가 생성한 HPA는 다음 값을 관측했습니다.

- external metric: **500m**
- CI target: **400m**
- replicas: **2 → 3 desired**
- ScaledObject Ready: **True**

운영용 `k8s/keda-scaledobject.yaml`의 threshold 0.60은 변경하지 않았습니다. CI workload가 약 0.50에서 안정화되므로 workflow에서 생성된 live ScaledObject만 0.40으로 패치했습니다.

artifact ID: **11406358252**

artifact digest: `sha256:9a1deeaf90f738bb38d7cfddcef9a1c08fac4061f11d5c4fc78fc1c8003550eb`
