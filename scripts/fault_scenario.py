from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx


async def run(base_url: str, *, failures: int, recovery_wait_s: float) -> dict:
    events: list[dict] = []
    async with httpx.AsyncClient(timeout=3) as client:
        for index in range(failures):
            response = await client.post(
                f"{base_url}/v1/predict",
                json={"text": f"fault-{index} __FAIL_PRIMARY__"},
            )
            body = response.json()
            events.append(
                {
                    "step": f"forced-failure-{index + 1}",
                    "status": response.status_code,
                    "backend": body.get("backend"),
                    "fallback_used": body.get("fallback_used"),
                }
            )

        opened = (await client.get(f"{base_url}/ops/status")).json()
        while_open = await client.post(
            f"{base_url}/v1/predict",
            json={"text": "healthy request while breaker is open"},
        )
        while_open_body = while_open.json()
        events.append(
            {
                "step": "healthy-while-open",
                "status": while_open.status_code,
                "backend": while_open_body.get("backend"),
                "fallback_used": while_open_body.get("fallback_used"),
            }
        )

        await asyncio.sleep(recovery_wait_s)
        recovered_response = await client.post(
            f"{base_url}/v1/predict",
            json={"text": "healthy request after recovery window"},
        )
        recovered_body = recovered_response.json()
        recovered_status = (await client.get(f"{base_url}/ops/status")).json()
        events.append(
            {
                "step": "recovery-probe",
                "status": recovered_response.status_code,
                "backend": recovered_body.get("backend"),
                "fallback_used": recovered_body.get("fallback_used"),
            }
        )

    ok = (
        opened["breaker_state"] == "open"
        and while_open.status_code == 200
        and while_open_body.get("fallback_used") is True
        and recovered_response.status_code == 200
        and recovered_body.get("fallback_used") is False
        and recovered_status["breaker_state"] == "closed"
    )
    return {
        "passed": ok,
        "opened_status": opened,
        "recovered_status": recovered_status,
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--failures", type=int, default=3)
    parser.add_argument("--recovery-wait-s", type=float, default=5.2)
    parser.add_argument("--output")
    args = parser.parse_args()

    result = asyncio.run(
        run(args.url, failures=args.failures, recovery_wait_s=args.recovery_wait_s)
    )
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
