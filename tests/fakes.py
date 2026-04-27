"""Test fakes for protocols. Deterministic, no I/O."""

from __future__ import annotations

import hashlib

from src.types import Query, RetrievalResult


class FakeEmbedder:
    """Hash-based deterministic embedder. Embeds texts to dim-D vectors."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self._dim)]


class FakeRetriever:
    """Retriever that returns a fixed list of results, ignoring the query."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        return self._results[: query.top_k]
