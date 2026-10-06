from __future__ import annotations

from typing import Any

import httpx

from app.backends.base import InferenceBackend, ModelResult


class TritonProtocolError(RuntimeError):
    pass


class TritonHTTPBackend(InferenceBackend):
    """Minimal NVIDIA Triton V2 HTTP adapter for a batched text classifier.

    Expected model contract:
    - BYTES input named TEXT (configurable), shape [batch, 1]
    - LABEL output with one string per item
    - SCORE output with one float per item

    The gateway keeps this adapter deliberately small so the reliability layer is
    independent from a particular model implementation or framework.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        input_name: str = "TEXT",
        label_output: str = "LABEL",
        score_output: str = "SCORE",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.input_name = input_name
        self.label_output = label_output
        self.score_output = score_output
        self.name = f"triton:{model}"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=None)

    async def infer_batch(self, texts: list[str]) -> list[ModelResult]:
        payload = {
            "inputs": [
                {
                    "name": self.input_name,
                    "shape": [len(texts), 1],
                    "datatype": "BYTES",
                    "data": texts,
                }
            ],
            "outputs": [
                {"name": self.label_output},
                {"name": self.score_output},
            ],
        }
        response = await self._client.post(
            f"{self.base_url}/v2/models/{self.model}/infer",
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        outputs = self._index_outputs(body)
        labels = outputs[self.label_output]
        scores = outputs[self.score_output]
        if len(labels) != len(texts) or len(scores) != len(texts):
            raise TritonProtocolError("Triton output length does not match input batch")

        return [
            ModelResult(
                label=str(label),
                score=float(score),
                backend=self.name,
            )
            for label, score in zip(labels, scores, strict=True)
        ]

    def _index_outputs(self, body: dict[str, Any]) -> dict[str, list[Any]]:
        raw_outputs = body.get("outputs")
        if not isinstance(raw_outputs, list):
            raise TritonProtocolError("Triton response is missing outputs")

        indexed: dict[str, list[Any]] = {}
        for output in raw_outputs:
            if not isinstance(output, dict):
                continue
            name = output.get("name")
            data = output.get("data")
            if isinstance(name, str) and isinstance(data, list):
                indexed[name] = data

        missing = [
            name
            for name in (self.label_output, self.score_output)
            if name not in indexed
        ]
        if missing:
            raise TritonProtocolError(f"Triton response missing outputs: {', '.join(missing)}")
        return indexed

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
