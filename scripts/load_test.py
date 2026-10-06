from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter

import httpx


async def run(base_url: str, requests: int, concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    codes: Counter[int] = Counter()
    batches: list[int] = []

    async with httpx.AsyncClient(timeout=10) as client:
        async def one(i: int) -> None:
            async with sem:
                start = time.perf_counter()
                response = await client.post(f"{base_url}/v1/predict", json={"text": f"request-{i}"})
                latencies.append((time.perf_counter() - start) * 1000)
                codes[response.status_code] += 1
                if response.status_code == 200:
                    batches.append(response.json()["batch_size"])

        started = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(requests)))
        duration = time.perf_counter() - started

    ordered = sorted(latencies)
    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
        return ordered[idx]

    return {
        "requests": requests,
        "concurrency": concurrency,
        "duration_s": round(duration, 3),
        "throughput_rps": round(requests / duration, 2),
        "success_rate": round(codes[200] / requests, 6),
        "status_codes": dict(codes),
        "p50_ms": round(pct(0.50), 2),
        "p95_ms": round(pct(0.95), 2),
        "p99_ms": round(pct(0.99), 2),
        "mean_batch_size": round(statistics.mean(batches), 2) if batches else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=400)
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = asyncio.run(run(args.url, args.requests, args.concurrency))
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fp:
            fp.write(payload + "\n")


if __name__ == "__main__":
    main()
