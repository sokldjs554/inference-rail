from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelResult:
    label: str
    score: float
    backend: str
    fallback_used: bool = False


class InferenceBackend(ABC):
    @abstractmethod
    async def infer_batch(self, texts: list[str]) -> list[ModelResult]:
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release backend resources when the gateway shuts down."""
        return None
