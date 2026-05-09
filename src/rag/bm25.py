"""BM25 sparse retrieval over Chunk text. Backed by `rank_bm25.BM25Okapi`."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.types import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Bm25Hit:
    """A scored BM25 hit."""

    chunk_id: str
    score: float


class Bm25Index:
    """Mutable BM25 index. `add()` invalidates the cached BM25 model; rebuilt on search."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._tokenized: list[list[str]] = []
        self._model: BM25Okapi | None = None

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks.append(chunk)
            self._tokenized.append(_tokenize(chunk.indexed_text))
        self._model = None

    def _ensure_model(self) -> BM25Okapi | None:
        if not self._chunks:
            return None
        if self._model is None:
            self._model = BM25Okapi(self._tokenized)
        return self._model

    def search(self, query: str, top_k: int, *, paper_filter: str | None = None) -> list[Bm25Hit]:
        """Top-`top_k` BM25 hits, optionally filtered to one paper.

        `paper_filter` matches `Chunk.paper_id` exactly. Used by the eval-side
        paper-id-aware retrieval path (ADR 0009 follow-up) to scope queries
        whose origin paper is known in the golden labels. Production callers
        pass `None` (no filter) since they have no paper hint.
        """
        model = self._ensure_model()
        if model is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = model.get_scores(tokens)
        if paper_filter is not None:
            allowed = [i for i, c in enumerate(self._chunks) if c.paper_id == paper_filter]
        else:
            allowed = list(range(len(self._chunks)))
        ranked = sorted(allowed, key=lambda i: scores[i], reverse=True)[:top_k]
        return [Bm25Hit(chunk_id=self._chunks[i].chunk_id, score=float(scores[i])) for i in ranked]
