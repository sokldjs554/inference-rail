# GPU runtime benchmark

현재 저장소에서 실제로 남은 검증 범위는 GPU입니다.

`scripts/gpu_runtime_benchmark.py`는 NVIDIA GPU가 연결된 환경에서 InferenceRail에 실제 부하를 보내면서 같은 시간대의 `nvidia-smi` 지표를 수집합니다.

## 전제

- NVIDIA GPU / driver / `nvidia-smi`
- InferenceRail gateway가 실행 중
- `BACKEND_MODE=triton`
- 연결된 Triton model이 **실제로 GPU에서 실행되는 모델**
- 공개 CPU Python backend 모델을 GPU 성능 측정에 사용하지 않음

## 실행

```bash
python scripts/gpu_runtime_benchmark.py \
  --url http://127.0.0.1:8000 \
  --requests 512 \
  --concurrency 64 \
  --output gpu-runtime-evidence.json
```

수집 항목:

- throughput req/s
- success rate
- p50 / p95 / p99 latency
- mean / max batch size
- backend distribution / fallback count
- GPU utilization average / max
- GPU memory utilization
- peak VRAM
- GPU power average / max
- GPU name / driver / total VRAM

GPU가 없는 환경에서는 명시적으로 실패합니다. 따라서 GPU evidence JSON이 실제로 생성되기 전까지 README나 포트폴리오에서 GPU 성능 수치를 주장하지 않습니다.
