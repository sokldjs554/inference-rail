from __future__ import annotations

import hashlib

from fastapi import FastAPI, HTTPException

app = FastAPI(title="Triton V2 contract stub")


@app.get("/health/ready")
async def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/v2/models/{model}/infer")
async def infer(model: str, payload: dict) -> dict:
    try:
        raw = payload["inputs"][0]["data"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="invalid Triton V2 input payload") from exc
    if not isinstance(raw, list) or not all(isinstance(text, str) for text in raw):
        raise HTTPException(status_code=400, detail="TEXT data must be a string list")

    labels: list[str] = []
    scores: list[float] = []
    for text in raw:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        score = int.from_bytes(digest, "big") / (2**64 - 1)
        scores.append(round(score, 6))
        labels.append("review" if score >= 0.62 else "pass")

    return {
        "model_name": model,
        "outputs": [
            {"name": "LABEL", "datatype": "BYTES", "shape": [len(raw)], "data": labels},
            {"name": "SCORE", "datatype": "FP32", "shape": [len(raw)], "data": scores},
        ],
    }
