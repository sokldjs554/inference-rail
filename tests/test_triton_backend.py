import httpx
import pytest

from app.backends.triton import TritonHTTPBackend


@pytest.mark.asyncio
async def test_triton_http_backend_batches_texts_and_parses_outputs():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        body = __import__("json").loads(request.content)
        seen["body"] = body
        return httpx.Response(
            200,
            json={
                "model_name": "text_classifier",
                "outputs": [
                    {
                        "name": "LABEL",
                        "datatype": "BYTES",
                        "shape": [2],
                        "data": ["pass", "review"],
                    },
                    {"name": "SCORE", "datatype": "FP32", "shape": [2], "data": [0.2, 0.91]},
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = TritonHTTPBackend(
        base_url="http://triton.test:8000",
        model="text_classifier",
        client=client,
    )
    try:
        result = await backend.infer_batch(["a", "b"])
    finally:
        await client.aclose()

    assert seen["path"] == "/v2/models/text_classifier/infer"
    assert seen["body"]["inputs"][0]["data"] == ["a", "b"]
    assert [item.label for item in result] == ["pass", "review"]
    assert [item.score for item in result] == [0.2, 0.91]
    assert all(item.backend == "triton:text_classifier" for item in result)
