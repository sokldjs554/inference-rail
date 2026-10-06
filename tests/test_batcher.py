import asyncio

import pytest

from app.backends.mock import MockBackend
from app.core.batcher import DynamicBatcher, QueueFullError


@pytest.mark.asyncio
async def test_dynamic_batching_groups_concurrent_requests():
    backend = MockBackend("test", base_latency_ms=4, item_latency_ms=1)
    batcher = DynamicBatcher(backend, max_batch_size=4, max_wait_ms=20, capacity=20)
    await batcher.start()
    try:
        results = await asyncio.gather(*(batcher.submit(f"item-{i}") for i in range(4)))
        assert {r.batch_size for r in results} == {4}
        assert batcher.processed_requests == 4
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_bounded_queue_rejects_overflow():
    backend = MockBackend("slow", base_latency_ms=100, item_latency_ms=1)
    batcher = DynamicBatcher(backend, max_batch_size=1, max_wait_ms=0, capacity=1)
    # Deliberately do not start worker; first request occupies the queue.
    first = asyncio.create_task(batcher.submit("one"))
    await asyncio.sleep(0)
    with pytest.raises(QueueFullError):
        await batcher.submit("two")
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

@pytest.mark.asyncio
async def test_stop_drains_inflight_request():
    backend = MockBackend("slow", base_latency_ms=30, item_latency_ms=1)
    batcher = DynamicBatcher(backend, max_batch_size=1, max_wait_ms=0, capacity=4)
    await batcher.start()
    pending = asyncio.create_task(batcher.submit("finish-me"))
    await asyncio.sleep(0.005)
    await batcher.stop()
    outcome = await pending
    assert outcome.result.backend == "slow"
    assert batcher.is_running is False


@pytest.mark.asyncio
async def test_expired_queued_request_is_not_sent_to_backend():
    backend = MockBackend("slow", base_latency_ms=70, item_latency_ms=1)
    batcher = DynamicBatcher(backend, max_batch_size=1, max_wait_ms=0, capacity=4)
    await batcher.start()
    try:
        first = asyncio.create_task(batcher.submit("first", timeout_ms=200))
        await asyncio.sleep(0.005)
        with pytest.raises(TimeoutError):
            await batcher.submit("expires-in-queue", timeout_ms=10)
        await first
        await asyncio.sleep(0.01)
        assert batcher.processed_requests == 1
        assert batcher.expired_requests == 1
    finally:
        await batcher.stop()
