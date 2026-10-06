#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
EVIDENCE_DIR="docs/evidence/runtime"
mkdir -p "$EVIDENCE_DIR"
: > "$EVIDENCE_DIR/summary.txt"

log() {
  printf '%s\n' "$*" | tee -a "$EVIDENCE_DIR/summary.txt"
}

cleanup_load() {
  kubectl delete pod hpa-load keda-load --ignore-not-found --wait=false >/dev/null 2>&1 || true
}
trap cleanup_load EXIT

log "=== cluster ==="
kubectl version
kubectl get nodes -o wide | tee "$EVIDENCE_DIR/nodes.txt"

log "=== app rollout ==="
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/pdb.yaml
kubectl set image deployment/inference-rail app=inference-rail:k8s
kubectl patch deployment inference-rail --type=json -p='[
  {"op":"replace","path":"/spec/template/spec/containers/0/imagePullPolicy","value":"Never"}
]'
kubectl rollout status deployment/inference-rail --timeout=180s
kubectl get pods -l app=inference-rail -o wide | tee "$EVIDENCE_DIR/rollout-pods.txt"

kubectl port-forward svc/inference-rail 18000:80 >"$EVIDENCE_DIR/port-forward.log" 2>&1 &
PF_PID=$!
trap 'kill "$PF_PID" >/dev/null 2>&1 || true; cleanup_load' EXIT
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:18000/health/ready >"$EVIDENCE_DIR/health.json"; then
    break
  fi
  sleep 1
done
curl -fsS http://127.0.0.1:18000/health/ready
curl -fsS http://127.0.0.1:18000/ops/build | tee "$EVIDENCE_DIR/build.json"
curl -fsS http://127.0.0.1:18000/v1/predict \
  -H 'content-type: application/json' \
  -d '{"text":"kind-runtime-smoke"}' | tee "$EVIDENCE_DIR/predict.json"

python - <<'PY'
import json, os
build=json.load(open("docs/evidence/runtime/build.json"))
pred=json.load(open("docs/evidence/runtime/predict.json"))
assert build["revision"] == os.environ["GIT_SHA"]
assert pred["backend"] == "primary-model"
assert pred["fallback_used"] is False
print("kind rollout smoke passed")
PY

log "=== metrics-server + HPA actual scale-out ==="
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.9.0/components.yaml
kubectl patch deployment metrics-server -n kube-system --type=json -p='[
  {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}
]'
kubectl rollout status deployment/metrics-server -n kube-system --timeout=180s

for _ in $(seq 1 90); do
  if kubectl top pods >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
kubectl top pods | tee "$EVIDENCE_DIR/top-before-hpa.txt"

kubectl apply -f k8s/hpa.yaml
kubectl patch hpa inference-rail --type=merge -p='{"spec":{"metrics":[{"type":"Resource","resource":{"name":"cpu","target":{"type":"Utilization","averageUtilization":5}}}]}}'

kubectl run hpa-load --image=curlimages/curl:8.12.1 --restart=Never --command -- sh -c '
for w in $(seq 1 32); do
  (
    while true; do
      curl -sS -o /dev/null -X POST http://inference-rail/v1/predict \
        -H "content-type: application/json" \
        -d "{\"text\":\"hpa-load-$w\"}" || true
    done
  ) &
done
wait
'

HPA_SCALED=0
for _ in $(seq 1 36); do
  kubectl get hpa inference-rail -o wide | tee "$EVIDENCE_DIR/hpa-live.txt"
  replicas="$(kubectl get deployment inference-rail -o jsonpath='{.spec.replicas}')"
  if [ "$replicas" -gt 2 ]; then
    HPA_SCALED=1
    break
  fi
  sleep 5
done
kubectl delete pod hpa-load --ignore-not-found --wait=false >/dev/null
if [ "$HPA_SCALED" -ne 1 ]; then
  log "HPA failed to scale above 2 replicas"
  kubectl describe hpa inference-rail | tee "$EVIDENCE_DIR/hpa-describe-failure.txt"
  exit 1
fi
log "HPA scaled deployment above 2 replicas"
kubectl get deployment inference-rail -o wide | tee "$EVIDENCE_DIR/hpa-scaled-deployment.txt"
kubectl describe hpa inference-rail | tee "$EVIDENCE_DIR/hpa-describe.txt"

log "=== KEDA + Prometheus actual queue scale-out ==="
kubectl delete hpa inference-rail --ignore-not-found
kubectl scale deployment inference-rail --replicas=2
kubectl patch configmap inference-rail-config --type=merge -p='{
  "data":{
    "QUEUE_CAPACITY":"16",
    "MAX_BATCH_SIZE":"1",
    "MAX_BATCH_WAIT_MS":"0",
    "MOCK_BASE_LATENCY_MS":"250",
    "MOCK_ITEM_LATENCY_MS":"2",
    "REQUEST_TIMEOUT_MS":"1500"
  }
}'
kubectl rollout restart deployment/inference-rail
kubectl rollout status deployment/inference-rail --timeout=180s

kubectl apply -f k8s/ci-prometheus.yaml
kubectl rollout status deployment/prometheus -n monitoring --timeout=180s

helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm install keda kedacore/keda --version 2.21.0 -n keda --create-namespace --wait --timeout 5m
kubectl apply -f k8s/keda-scaledobject.yaml

for _ in $(seq 1 60); do
  ready="$(kubectl get scaledobject inference-rail-queue -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)"
  if [ "$ready" = "True" ]; then
    break
  fi
  sleep 2
done
kubectl get scaledobject inference-rail-queue -o yaml > "$EVIDENCE_DIR/scaledobject-ready.yaml"

kubectl run keda-load --image=curlimages/curl:8.12.1 --restart=Never --command -- sh -c '
for w in $(seq 1 80); do
  (
    while true; do
      curl -sS -o /dev/null -X POST http://inference-rail/v1/predict \
        -H "content-type: application/json" \
        -d "{\"text\":\"keda-load-$w\"}" || true
    done
  ) &
done
wait
'

KEDA_SCALED=0
for _ in $(seq 1 48); do
  replicas="$(kubectl get deployment inference-rail -o jsonpath='{.spec.replicas}')"
  queue_metric="$(kubectl -n monitoring exec deploy/prometheus -- \
    wget -qO- 'http://127.0.0.1:9090/api/v1/query?query=sum(inference_queue_depth)%2Fclamp_min(sum(inference_queue_capacity)%2C1)' \
    2>/dev/null || true)"
  printf '%s\n' "$queue_metric" > "$EVIDENCE_DIR/prometheus-queue-query.json"
  kubectl get hpa -o wide | tee "$EVIDENCE_DIR/keda-hpa-live.txt"
  if [ "$replicas" -gt 2 ]; then
    KEDA_SCALED=1
    break
  fi
  sleep 5
done
kubectl delete pod keda-load --ignore-not-found --wait=false >/dev/null
if [ "$KEDA_SCALED" -ne 1 ]; then
  log "KEDA failed to scale above 2 replicas"
  kubectl describe scaledobject inference-rail-queue | tee "$EVIDENCE_DIR/keda-describe-failure.txt"
  kubectl get hpa -o yaml > "$EVIDENCE_DIR/keda-hpa-failure.yaml" || true
  exit 1
fi
log "KEDA scaled deployment above 2 replicas from Prometheus queue saturation"
kubectl get deployment inference-rail -o wide | tee "$EVIDENCE_DIR/keda-scaled-deployment.txt"
kubectl describe scaledobject inference-rail-queue | tee "$EVIDENCE_DIR/keda-describe.txt"
kubectl get hpa -o yaml > "$EVIDENCE_DIR/keda-hpa.yaml"

log "Kubernetes runtime verification passed"
