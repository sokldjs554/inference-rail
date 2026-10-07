from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "triton-service-boundary-runtime-evidence.json"
PROXY_PORT = 18083
GATEWAY_PORT = 18091


async def wait_ready(url: str, attempts: int = 100) -> None:
    async with httpx.AsyncClient(timeout=1) as client:
        for _ in range(attempts):
            try:
                if (await client.get(url)).status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.2)
    raise RuntimeError(f"endpoint did not become ready: {url}")


async def post_predict(client: httpx.AsyncClient, text: str) -> dict:
    response = await client.post(
        f"http://127.0.0.1:{GATEWAY_PORT}/v1/predict",
        json={"text": text},
    )
    response.raise_for_status()
    return response.json()


async def verify() -> dict:
    async with httpx.AsyncClient(timeout=5) as client:
        healthy = await post_predict(client, "real-triton-boundary-healthy")
        assert healthy["backend"] == "triton:text_classifier"
        assert healthy["fallback_used"] is False

        slow = await post_predict(client, "__PROXY_SLOW__ real-triton-slow")
        assert slow["fallback_used"] is True
        status_after_slow = (
            await client.get(f"http://127.0.0.1:{GATEWAY_PORT}/ops/status")
        ).json()
        assert status_after_slow["primary_timeouts"] >= 1

        await asyncio.sleep(0.3)
        recovered_after_slow = await post_predict(client, "recover-after-slow")
        assert recovered_after_slow["backend"] == "triton:text_classifier"
        assert recovered_after_slow["fallback_used"] is False

        await client.post(f"http://127.0.0.1:{PROXY_PORT}/reset")
        failures = [
            await post_predict(client, f"__PROXY_FAIL__ hard-failure-{index}")
            for index in range(3)
        ]
        assert all(item["fallback_used"] for item in failures)
        open_status = (
            await client.get(f"http://127.0.0.1:{GATEWAY_PORT}/ops/status")
        ).json()
        assert open_status["breaker_state"] == "open"

        stats_before = (
            await client.get(f"http://127.0.0.1:{PROXY_PORT}/stats")
        ).json()
        while_open = await post_predict(client, "healthy-while-open")
        stats_after = (
            await client.get(f"http://127.0.0.1:{PROXY_PORT}/stats")
        ).json()
        assert while_open["fallback_used"] is True
        assert stats_after["inference_requests"] == stats_before["inference_requests"]

        await asyncio.sleep(0.3)
        recovery = await post_predict(client, "recovery-probe")
        final_status = (
            await client.get(f"http://127.0.0.1:{GATEWAY_PORT}/ops/status")
        ).json()
        final_stats = (
            await client.get(f"http://127.0.0.1:{PROXY_PORT}/stats")
        ).json()
        assert recovery["backend"] == "triton:text_classifier"
        assert recovery["fallback_used"] is False
        assert final_status["breaker_state"] == "closed"

    return {
        "passed": True,
        "path": "InferenceRail -> chaos proxy -> real NVIDIA Triton Server",
        "healthy": healthy,
        "slow_primary": {
            "response": slow,
            "runtime_status": status_after_slow,
        },
        "recovered_after_slow": recovered_after_slow,
        "hard_failure": {
            "responses": failures,
            "breaker_open_status": open_status,
            "proxy_requests_before_open_probe": stats_before["inference_requests"],
            "proxy_requests_after_open_probe": stats_after["inference_requests"],
            "healthy_while_open": while_open,
        },
        "recovery": {
            "response": recovery,
            "runtime_status": final_status,
            "proxy_stats": final_stats,
        },
    }


def terminate(process: subprocess.Popen[bytes]) -> None:
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> None:
    proxy_env = os.environ.copy()
    proxy_env.update(
        {
            "PYTHONPATH": str(ROOT),
            "TRITON_UPSTREAM": "http://127.0.0.1:18080",
            "TRITON_PROXY_SLOW_MS": "600",
        }
    )
    proxy = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "scripts.triton_chaos_proxy:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PROXY_PORT),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=proxy_env,
    )

    gateway_env = os.environ.copy()
    gateway_env.update(
        {
            "PYTHONPATH": str(ROOT),
            "APP_ENV": "github-actions-triton-boundary",
            "BACKEND_MODE": "triton",
            "TRITON_URL": f"http://127.0.0.1:{PROXY_PORT}",
            "TRITON_MODEL": "text_classifier",
            "MAX_BATCH_SIZE": "1",
            "MAX_BATCH_WAIT_MS": "2",
            "PRIMARY_TIMEOUT_MS": "200",
            "FALLBACK_TIMEOUT_MS": "200",
            "REQUEST_TIMEOUT_MS": "1200",
            "BREAKER_FAILURE_THRESHOLD": "3",
            "BREAKER_RECOVERY_SECONDS": "0.2",
            "GIT_SHA": os.environ["GIT_SHA"],
        }
    )
    gateway = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(GATEWAY_PORT),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=gateway_env,
    )

    try:
        asyncio.run(wait_ready(f"http://127.0.0.1:{PROXY_PORT}/healthz"))
        asyncio.run(wait_ready(f"http://127.0.0.1:{GATEWAY_PORT}/health/ready"))
        result = asyncio.run(verify())
    finally:
        terminate(gateway)
        terminate(proxy)

    EVIDENCE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
