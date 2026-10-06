from __future__ import annotations

import time
from typing import Any


class FlightRecorder:
    def __init__(self, max_flights: int = 200) -> None:
        self.max_flights = max_flights
        self._flights: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []

    def start(self, request_id: str, text: str) -> None:
        now = time.time()
        self._flights[request_id] = {
            "request_id": request_id,
            "started_at": now,
            "started_monotonic": time.monotonic(),
            "preview": text[:96],
            "status": "running",
            "events": [],
        }
        self._order.append(request_id)
        while len(self._order) > self.max_flights:
            expired = self._order.pop(0)
            self._flights.pop(expired, None)
        self.record(request_id, "received", {"preview": text[:96]})

    def record(
        self,
        request_id: str,
        stage: str,
        detail: dict[str, object] | None = None,
    ) -> None:
        flight = self._flights.get(request_id)
        if flight is None:
            return
        elapsed_ms = (time.monotonic() - flight["started_monotonic"]) * 1000
        flight["events"].append(
            {
                "stage": stage,
                "at_ms": round(elapsed_ms, 2),
                "detail": detail or {},
            }
        )

    def finish(self, request_id: str, status: str) -> None:
        flight = self._flights.get(request_id)
        if flight is None:
            return
        flight["status"] = status
        flight["duration_ms"] = round(
            (time.monotonic() - flight["started_monotonic"]) * 1000,
            2,
        )

    def get(self, request_id: str) -> dict[str, Any] | None:
        flight = self._flights.get(request_id)
        if flight is None:
            return None
        return self._public(flight)

    def recent(self, limit: int = 12) -> list[dict[str, Any]]:
        ids = self._order[-max(1, min(limit, 50)) :]
        return [self._public(self._flights[request_id]) for request_id in reversed(ids)]

    @staticmethod
    def _public(flight: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in flight.items()
            if key != "started_monotonic"
        }
