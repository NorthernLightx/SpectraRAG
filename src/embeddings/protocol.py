"""Embedder Protocol: the only embedding interface upstream code depends on."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Embed text into dense vectors."""

    @property
    def dim(self) -> int:
        """Embedding dimensionality. Stable for a given model identity."""
        ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors. Returns one vector per input."""
        ...
