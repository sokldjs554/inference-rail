from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.trace import Link, SpanContext

from app.backends.base import InferenceBackend, ModelResult

_TRACER = trace.get_tracer(__name__)


class QueueFullError(RuntimeError):
    pass


@dataclass
class BatchOutcome:
    result: ModelResult
    batch_size: int
    queue_ms: float
    service_ms: float


@dataclass
class _Item:
    text: str
    enqueued_at: float
    deadline_at: float | None
    future: asyncio.Future[BatchOutcome]
    span_context: SpanContext | None


class DynamicBatcher:
    def __init__(
        self,
        backend: InferenceBackend,
        *,
        max_batch_size: int,
        max_wait_ms: int,
        capacity: int,
        drain_timeout_ms: int = 5000,
    ) -> None:
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000
        self.queue: asyncio.Queue[_Item] = asyncio.Queue(maxsize=capacity)
        self.drain_timeout_s = drain_timeout_ms / 1000
        self._worker: asyncio.Task[None] | None = None
        self.processed_requests = 0
        self.expired_requests = 0

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()

    @property
    def capacity(self) -> int:
        return self.queue.maxsize

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="dynamic-batcher")

    @property
    def is_running(self) -> bool:
        return self._worker is not None and not self._worker.done()

    async def stop(self) -> None:
        if self._worker is None:
            await self.backend.aclose()
            return
        # Queue.join() also waits for items already removed from the queue but still
        # being processed, so in-flight inference is drained before worker cancellation.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.queue.join(), timeout=self.drain_timeout_s)
        self._worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker
        self._worker = None
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if not item.future.done():
                item.future.cancel()
            self.queue.task_done()
        await self.backend.aclose()

    async def submit(self, text: str, *, timeout_ms: int | None = None) -> BatchOutcome:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[BatchOutcome] = loop.create_future()
        now = time.monotonic()
        deadline_at = None if timeout_ms is None else now + timeout_ms / 1000
        current_span_context = trace.get_current_span().get_span_context()
        item = _Item(
            text=text,
            enqueued_at=now,
            deadline_at=deadline_at,
            future=future,
            span_context=current_span_context if current_span_context.is_valid else None,
        )
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            raise QueueFullError("inference queue is full") from exc

        if timeout_ms is None:
            return await future

        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout_ms / 1000)
        except TimeoutError:
            future.cancel()
            raise

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            now = time.monotonic()
            batch_window_deadline = now + self.max_wait_s
            if first.deadline_at is not None:
                batch_window_deadline = min(batch_window_deadline, first.deadline_at)

            while len(batch) < self.max_batch_size:
                remaining = batch_window_deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch.append(item)
                    if item.deadline_at is not None:
                        batch_window_deadline = min(batch_window_deadline, item.deadline_at)
                except TimeoutError:
                    break

            active: list[_Item] = []
            current = time.monotonic()
            for item in batch:
                if item.future.cancelled() or (
                    item.deadline_at is not None and current >= item.deadline_at
                ):
                    self.expired_requests += 1
                    if not item.future.done():
                        item.future.set_exception(
                            TimeoutError("inference request deadline exceeded")
                        )
                    self.queue.task_done()
                else:
                    active.append(item)

            if not active:
                continue

            started = time.monotonic()
            links = [Link(item.span_context) for item in active if item.span_context is not None]
            try:
                with _TRACER.start_as_current_span("inference.batch", links=links) as span:
                    span.set_attribute("inference.batch.size", len(active))
                    span.set_attribute("inference.queue.depth_after_dequeue", self.queue.qsize())
                    results = await self.backend.infer_batch([item.text for item in active])
                    service_ms = (time.monotonic() - started) * 1000
                    span.set_attribute("inference.batch.service_ms", service_ms)
                    if len(results) != len(active):
                        raise RuntimeError("backend returned unexpected result count")
                    for item, result in zip(active, results, strict=True):
                        if not item.future.cancelled():
                            item.future.set_result(
                                BatchOutcome(
                                    result=result,
                                    batch_size=len(active),
                                    queue_ms=(started - item.enqueued_at) * 1000,
                                    service_ms=service_ms,
                                )
                            )
                        self.processed_requests += 1
            except asyncio.CancelledError:
                for item in active:
                    if not item.future.done():
                        item.future.cancel()
                raise
            except Exception as exc:
                for item in active:
                    if not item.future.cancelled():
                        item.future.set_exception(exc)
            finally:
                for _ in active:
                    self.queue.task_done()
