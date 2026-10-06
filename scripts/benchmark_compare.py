from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


async def wait_ready(port: int) -> None:
    async with httpx.AsyncClient() as client:
        for _ in range(80):
            try:
                response = await client.get(
                    f"http://127.0.0.1:{port}/health/ready", timeout=0.4
                )
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.1)
    raise RuntimeError("server did not become ready")


async def run_load(port: int, count: int, concurrency: int) -> dict:
    from scripts.load_test import run
    return await run(f"http://127.0.0.1:{port}", count, concurrency)


def run_case(batch_size: int, port: int, count: int, concurrency: int) -> dict:
    env = os.environ.copy()
    env.update({
        "MAX_BATCH_SIZE": str(batch_size),
        "MAX_BATCH_WAIT_MS": "10" if batch_size > 1 else "0",
        "PYTHONPATH": str(ROOT),
    })
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=env,
    )
    try:
        asyncio.run(wait_ready(port))
        result = asyncio.run(run_load(port, count, concurrency))
        result["max_batch_size"] = batch_size
        return result
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=160)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--output", default="benchmark-results.json")
    args = parser.parse_args()

    baseline = run_case(1, 8091, args.requests, args.concurrency)
    batched = run_case(8, 8092, args.requests, args.concurrency)
    result = {
        "baseline": baseline,
        "batched": batched,
        "throughput_gain_x": round(batched["throughput_rps"] / baseline["throughput_rps"], 2),
        "p95_change_pct": round((batched["p95_ms"] / baseline["p95_ms"] - 1) * 100, 2),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
