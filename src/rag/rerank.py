"""Cross-encoder reranker. Default: BGE reranker v2 m3 via sentence-transformers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.types import Chunk

ScorerFn = Callable[[list[tuple[str, str]]], list[float]]
"""A callable that scores (query, document) pairs into floats. Higher = more relevant."""

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class RerankedHit:
    """A chunk re-scored by a cross-encoder."""

    chunk_id: str
    rerank_score: float


class BgeReranker:
    """BGE reranker. Loads sentence-transformers CrossEncoder lazily on first use."""

    def __init__(
        self,
        *,
        scorer: ScorerFn | None = None,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self._injected_scorer = scorer
        self._model_name = model_name
        self._ce: object | None = None

    def _resolve_scorer(self) -> ScorerFn:
        if self._injected_scorer is not None:
            return self._injected_scorer
        if self._ce is None:
            from sentence_transformers import CrossEncoder

            self._ce = CrossEncoder(self._model_name)
        ce = self._ce
        assert ce is not None  # narrow for mypy

        def _score(pairs: list[tuple[str, str]]) -> list[float]:
            raw = ce.predict(pairs)  # type: ignore[attr-defined]
            return [float(s) for s in raw]

        return _score

    def rerank(self, query: str, candidates: list[Chunk], top_k: int) -> list[RerankedHit]:
        """Score (query, chunk.text) pairs, sort descending, cap to top_k."""
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self._resolve_scorer()(pairs)
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [
            RerankedHit(chunk_id=chunk.chunk_id, rerank_score=float(score))
            for chunk, score in ranked[:top_k]
        ]
