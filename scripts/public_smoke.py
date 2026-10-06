from __future__ import annotations

import argparse
import asyncio
import json

import httpx


async def run(base_url: str) -> dict:
    base = base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        ready = await client.get(f"{base}/health/ready")
        build = await client.get(f"{base}/ops/build")
        prediction = await client.post(f"{base}/v1/predict", json={"text": "public-smoke"})
        status = await client.get(f"{base}/ops/status")

    result = {
        "passed": all(r.status_code == 200 for r in (ready, build, prediction, status)),
        "ready": ready.json(),
        "build": build.json(),
        "prediction": prediction.json(),
        "runtime_status": status.json(),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    args = parser.parse_args()
    result = asyncio.run(run(args.base_url))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
