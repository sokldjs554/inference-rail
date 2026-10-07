from __future__ import annotations

import asyncio
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

UPSTREAM = os.getenv("TRITON_UPSTREAM", "http://127.0.0.1:18080").rstrip("/")
SLOW_MS = int(os.getenv("TRITON_PROXY_SLOW_MS", "600"))

app = FastAPI(title="Triton Chaos Proxy")
stats = {"inference_requests": 0, "forwarded": 0, "failed": 0, "slow": 0}


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
async def get_stats() -> dict[str, int]:
    return dict(stats)


@app.post("/reset")
async def reset() -> dict[str, int]:
    for key in stats:
        stats[key] = 0
    return dict(stats)


@app.post("/v2/models/{model}/infer")
async def infer(model: str, request: Request) -> Response:
    body = await request.body()
    marker_text = body.decode("utf-8", errors="ignore")
    stats["inference_requests"] += 1

    if "__PROXY_FAIL__" in marker_text:
        stats["failed"] += 1
        return JSONResponse(status_code=503, content={"error": "injected proxy failure"})

    if "__PROXY_SLOW__" in marker_text:
        stats["slow"] += 1
        await asyncio.sleep(SLOW_MS / 1000)

    async with httpx.AsyncClient(timeout=10) as client:
        upstream = await client.post(
            f"{UPSTREAM}/v2/models/{model}/infer",
            content=body,
            headers={"content-type": "application/json"},
        )
    stats["forwarded"] += 1
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )
