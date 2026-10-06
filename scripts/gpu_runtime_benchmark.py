from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

import httpx


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def query_nvidia(fields: str) -> list[list[str]]:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi not found; an NVIDIA GPU runtime is required")
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[list[str]] = []
    for line in result.stdout.splitlines():
        if line.strip():
            rows.append([part.strip() for part in line.split(",")])
    return rows


def gpu_metadata() -> list[dict[str, str]]:
    rows = query_nvidia("index,name,driver_version,memory.total")
    return [
        {
            "index": row[0],
            "name": row[1],
            "driver_version": row[2],
            "memory_total_mib": row[3],
        }
        for row in rows
    ]


async def sample_gpu(
    samples: list[dict[str, float]],
    stop: asyncio.Event,
    interval_s: float,
) -> None:
    while not stop.is_set():
        try:
            rows = query_nvidia(
                "utilization.gpu,utilization.memory,memory.used,power.draw"
            )
            for row in rows:
                samples.append(
                    {
                        "gpu_util_pct": float(row[0]),
                        "memory_util_pct": float(row[1]),
                        "memory_used_mib": float(row[2]),
                        "power_w": float(row[3]),
                    }
                )
        except (ValueError, subprocess.CalledProcessError):
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except TimeoutError:
            pass


async def run_load(base_url: str, count: int, concurrency: int) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    statuses: Counter[int] = Counter()
    backends: Counter[str] = Counter()
    batch_sizes: list[int] = []
    fallbacks = 0

    async with httpx.AsyncClient(timeout=30) as client:

        async def one(index: int) -> None:
            nonlocal fallbacks
            async with semaphore:
                started = time.perf_counter()
                response = await client.post(
                    f"{base_url.rstrip('/')}/v1/predict",
                    json={"text": f"gpu-runtime-{index}"},
                )
                latencies.append((time.perf_counter() - started) * 1000)
                statuses[response.status_code] += 1
                if response.status_code == 200:
                    body = response.json()
                    backends[body["backend"]] += 1
                    batch_sizes.append(body["batch_size"])
                    fallbacks += int(body["fallback_used"])

        started = time.perf_counter()
        await asyncio.gather(*(one(index) for index in range(count)))
        duration = time.perf_counter() - started

    return {
        "requests": count,
        "concurrency": concurrency,
        "duration_s": round(duration, 3),
        "throughput_rps": round(count / duration, 2),
        "success_rate": round(statuses[200] / count, 6),
        "status_codes": dict(statuses),
        "p50_ms": round(percentile(latencies, 0.50), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "p99_ms": round(percentile(latencies, 0.99), 2),
        "mean_batch_size": round(statistics.mean(batch_sizes), 2)
        if batch_sizes
        else 0,
        "max_batch_size": max(batch_sizes, default=0),
        "backends": dict(backends),
        "fallback_requests": fallbacks,
    }


def summarize_gpu(samples: list[dict[str, float]]) -> dict:
    if not samples:
        return {"samples": 0}
    return {
        "samples": len(samples),
        "gpu_util_avg_pct": round(
            statistics.mean(item["gpu_util_pct"] for item in samples), 2
        ),
        "gpu_util_max_pct": max(item["gpu_util_pct"] for item in samples),
        "memory_used_max_mib": max(item["memory_used_mib"] for item in samples),
        "memory_util_max_pct": max(item["memory_util_pct"] for item in samples),
        "power_avg_w": round(
            statistics.mean(item["power_w"] for item in samples), 2
        ),
        "power_max_w": max(item["power_w"] for item in samples),
    }


async def execute(args: argparse.Namespace) -> dict:
    metadata = gpu_metadata()
    samples: list[dict[str, float]] = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(sample_gpu(samples, stop, args.sample_interval))
    try:
        load = await run_load(args.url, args.requests, args.concurrency)
    finally:
        stop.set()
        await sampler

    return {
        "passed": load["success_rate"] == 1.0 and load["fallback_requests"] == 0,
        "gpu": metadata,
        "load": load,
        "gpu_samples": summarize_gpu(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--output", default="gpu-runtime-evidence.json")
    args = parser.parse_args()

    result = asyncio.run(execute(args))
    payload = json.dumps(result, indent=2)
    Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
