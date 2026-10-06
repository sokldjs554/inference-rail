from __future__ import annotations

import asyncio
import hashlib

from app.backends.base import InferenceBackend, ModelResult


class MockBackend(InferenceBackend):
    """Deterministic backend used to benchmark server behavior without model downloads.

    Primary can be forced to fail by sending text containing ``__FAIL_PRIMARY__``.
    Latency models fixed model overhead plus sub-linear batch cost, making batching measurable.
    """

    def __init__(
        self,
        name: str,
        *,
        base_latency_ms: float,
        item_latency_ms: float,
        fail_on_marker: bool = False,
    ) -> None:
        self.name = name
        self.base_latency_ms = base_latency_ms
        self.item_latency_ms = item_latency_ms
        self.fail_on_marker = fail_on_marker

    async def infer_batch(self, texts: list[str]) -> list[ModelResult]:
        if self.fail_on_marker and any("__FAIL_PRIMARY__" in text for text in texts):
            await asyncio.sleep(0.004)
            raise RuntimeError("simulated primary model failure")

        batch_size = max(1, len(texts))
        latency_ms = self.base_latency_ms + self.item_latency_ms * (batch_size**0.72)
        await asyncio.sleep(latency_ms / 1000)

        results: list[ModelResult] = []
        for text in texts:
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big") / (2**64 - 1)
            score = round(raw, 6)
            results.append(
                ModelResult(
                    label="review" if score >= 0.62 else "pass",
                    score=score,
                    backend=self.name,
                )
            )
        return results
