from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import httpx

from scripts.fault_scenario import run as run_fault_scenario
from scripts.load_test import run as run_load
from scripts.validate_manifests import validate as validate_manifests

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "docs" / "evidence"


async def wait_ready(port: int) -> None:
    async with httpx.AsyncClient() as client:
        for _ in range(100):
            try:
                response = await client.get(
                    f"http://127.0.0.1:{port}/health/ready",
                    timeout=0.3,
                )
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.05)
    raise RuntimeError(f"server on port {port} did not become ready")


def start_server(
    port: int,
    extra_env: dict[str, str],
    *,
    module: str = "app.main:app",
) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(extra_env)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            module,
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


def stop_server(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=4)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def run_subprocess(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    evidence: dict = {
        "tests": {},
        "manifests": {},
        "triton_contract": {},
        "fault_scenario": {},
        "overload": {},
        "benchmark": {},
    }

    run_subprocess([sys.executable, "-m", "pytest", "-q"])
    evidence["tests"] = {"passed": True}
    evidence["manifests"] = validate_manifests()

    triton_stub_port = 8095
    triton_gateway_port = 8096
    triton_stub = start_server(
        triton_stub_port,
        {},
        module="scripts.triton_contract_stub:app",
    )
    triton_gateway = None
    try:
        asyncio.run(wait_ready(triton_stub_port))
        triton_gateway = start_server(
            triton_gateway_port,
            {
                "BACKEND_MODE": "triton",
                "TRITON_URL": f"http://127.0.0.1:{triton_stub_port}",
                "TRITON_MODEL": "text_classifier",
            },
        )
        asyncio.run(wait_ready(triton_gateway_port))

        async def exercise_triton_contract() -> dict:
            async with httpx.AsyncClient(timeout=3) as client:
                responses = await asyncio.gather(
                    *(
                        client.post(
                            f"http://127.0.0.1:{triton_gateway_port}/v1/predict",
                            json={"text": f"contract-{index}"},
                        )
                        for index in range(8)
                    )
                )
                bodies = [response.json() for response in responses]
                status = (
                    await client.get(f"http://127.0.0.1:{triton_gateway_port}/ops/status")
                ).json()
                passed = (
                    all(response.status_code == 200 for response in responses)
                    and all(body.get("backend") == "triton:text_classifier" for body in bodies)
                    and all(body.get("fallback_used") is False for body in bodies)
                    and status.get("backend_mode") == "triton"
                )
                return {
                    "passed": passed,
                    "requests": len(responses),
                    "backends": sorted({body.get("backend") for body in bodies}),
                    "batch_sizes": sorted({body.get("batch_size") for body in bodies}),
                    "runtime_status": status,
                }

        triton_result = asyncio.run(exercise_triton_contract())
        if not triton_result["passed"]:
            raise RuntimeError("Triton HTTP contract scenario failed")
        evidence["triton_contract"] = triton_result
    finally:
        if triton_gateway is not None:
            stop_server(triton_gateway)
        stop_server(triton_stub)

    fault_port = 8093
    fault_proc = start_server(
        fault_port,
        {
            "BREAKER_FAILURE_THRESHOLD": "3",
            "BREAKER_RECOVERY_SECONDS": "0.15",
            "PRIMARY_TIMEOUT_MS": "700",
        },
    )
    try:
        asyncio.run(wait_ready(fault_port))
        fault_result = asyncio.run(
            run_fault_scenario(
                f"http://127.0.0.1:{fault_port}",
                failures=3,
                recovery_wait_s=0.20,
            )
        )
        if not fault_result["passed"]:
            raise RuntimeError("fault scenario failed")
        evidence["fault_scenario"] = fault_result
    finally:
        stop_server(fault_proc)

    overload_port = 8094
    overload_proc = start_server(
        overload_port,
        {
            "QUEUE_CAPACITY": "2",
            "MAX_BATCH_SIZE": "1",
            "MAX_BATCH_WAIT_MS": "0",
            "MOCK_BASE_LATENCY_MS": "120",
            "MOCK_ITEM_LATENCY_MS": "1",
            "REQUEST_TIMEOUT_MS": "800",
        },
    )
    try:
        asyncio.run(wait_ready(overload_port))
        overload = asyncio.run(
            run_load(
                f"http://127.0.0.1:{overload_port}",
                requests=60,
                concurrency=40,
            )
        )
        codes = {str(k): v for k, v in overload["status_codes"].items()}
        overload["status_codes"] = codes
        overload["passed"] = int(codes.get("429", 0)) > 0 and int(codes.get("200", 0)) > 0
        if not overload["passed"]:
            raise RuntimeError("overload scenario did not show both admission and fast rejection")
        evidence["overload"] = overload
    finally:
        stop_server(overload_proc)

    benchmark_output = EVIDENCE_DIR / "release-benchmark.json"
    run_subprocess(
        [
            sys.executable,
            "scripts/benchmark_compare.py",
            "--requests",
            "160",
            "--concurrency",
            "24",
            "--output",
            str(benchmark_output),
        ]
    )
    run_subprocess([sys.executable, "scripts/slo_gate.py", str(benchmark_output)])
    evidence["benchmark"] = json.loads(benchmark_output.read_text(encoding="utf-8"))

    output = EVIDENCE_DIR / "release-verification.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "evidence": str(output.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
