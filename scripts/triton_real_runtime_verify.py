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
EVIDENCE = ROOT / "triton-runtime-evidence.json"
GATEWAY_PORT = 18090


async def wait_ready(url: str, attempts: int = 80) -> None:
    async with httpx.AsyncClient(timeout=1.0) as client:
        for _ in range(attempts):
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.25)
    raise RuntimeError(f"endpoint did not become ready: {url}")


async def direct_triton_batch() -> dict:
    payload = {
        "inputs": [
            {
                "name": "TEXT",
                "shape": [4, 1],
                "datatype": "BYTES",
                "data": ["direct-1", "direct-2", "direct-3", "direct-4"],
            }
        ],
        "outputs": [{"name": "LABEL"}, {"name": "SCORE"}],
    }
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            "http://127.0.0.1:18080/v2/models/text_classifier/infer",
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
    outputs = {item["name"]: item["data"] for item in body["outputs"]}
    assert len(outputs["LABEL"]) == 4
    assert len(outputs["SCORE"]) == 4
    return {
        "status": response.status_code,
        "model_name": body.get("model_name"),
        "labels": outputs["LABEL"],
        "scores": outputs["SCORE"],
    }


async def gateway_batch() -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        responses = await asyncio.gather(
            *(
                client.post(
                    f"http://127.0.0.1:{GATEWAY_PORT}/v1/predict",
                    json={"text": f"real-triton-{index}"},
                )
                for index in range(8)
            )
        )
        bodies = [response.json() for response in responses]
        status = (
            await client.get(f"http://127.0.0.1:{GATEWAY_PORT}/ops/status")
        ).json()
        build = (
            await client.get(f"http://127.0.0.1:{GATEWAY_PORT}/ops/build")
        ).json()

    assert all(response.status_code == 200 for response in responses)
    assert all(body["backend"] == "triton:text_classifier" for body in bodies)
    assert all(body["fallback_used"] is False for body in bodies)
    assert all(body["batch_size"] == 8 for body in bodies)
    assert status["backend_mode"] == "triton"
    assert status["fallback_requests"] == 0
    assert status["primary_failures"] == 0
    assert status["primary_timeouts"] == 0
    assert build["revision"] == os.environ["GIT_SHA"]

    return {
        "requests": len(responses),
        "backends": sorted({body["backend"] for body in bodies}),
        "batch_sizes": sorted({body["batch_size"] for body in bodies}),
        "fallback_used": any(body["fallback_used"] for body in bodies),
        "runtime_status": status,
        "build": build,
    }


def main() -> None:
    direct = asyncio.run(direct_triton_batch())

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "APP_ENV": "github-actions-real-triton",
            "BACKEND_MODE": "triton",
            "TRITON_URL": "http://127.0.0.1:18080",
            "TRITON_MODEL": "text_classifier",
            "MAX_BATCH_SIZE": "8",
            "MAX_BATCH_WAIT_MS": "25",
            "PRIMARY_TIMEOUT_MS": "3000",
            "REQUEST_TIMEOUT_MS": "5000",
        }
    )
    process = subprocess.Popen(
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
        env=env,
    )
    try:
        asyncio.run(
            wait_ready(f"http://127.0.0.1:{GATEWAY_PORT}/health/ready")
        )
        gateway = asyncio.run(gateway_batch())
    finally:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    result = {
        "passed": True,
        "triton_container": "nvcr.io/nvidia/tritonserver:26.09-py3",
        "triton_release": "2.73.0",
        "execution": "CPU Python backend in real Triton Server",
        "direct_triton_v2": direct,
        "gateway_to_real_triton": gateway,
    }
    EVIDENCE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
