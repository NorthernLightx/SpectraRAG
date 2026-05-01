"""Cross-encoder reranker. Default: BGE reranker v2 m3 via sentence-transformers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.types import Chunk

ScorerFn = Callable[[list[tuple[str, str]]], list[float]]
"""A callable that scores (query, document) pairs into floats. Higher = more relevant."""

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


def _autodetect_device() -> str:
    """Return 'cuda' if torch reports a CUDA device, else 'cpu'.

    Imported lazily because sentence-transformers (and torch) is a heavy dep that
    we don't want to load when an injected scorer is in use (e.g. tests).
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@dataclass(frozen=True)
class RerankedHit:
    """A chunk re-scored by a cross-encoder."""

    chunk_id: str
    rerank_score: float


class BgeReranker:
    """BGE reranker. Loads sentence-transformers CrossEncoder lazily on first use.

    `device` defaults to auto-detect (cuda if torch reports it, else cpu). Cross-encoder
    inference is ~40x faster on a consumer GPU vs CPU for this model.
    """

    def __init__(
        self,
        *,
        scorer: ScorerFn | None = None,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        self._injected_scorer = scorer
        self._model_name = model_name
        self._device = device
        self._ce: object | None = None

    def _resolve_scorer(self) -> ScorerFn:
        if self._injected_scorer is not None:
            return self._injected_scorer
        if self._ce is None:
            from sentence_transformers import CrossEncoder

            device = self._device if self._device is not None else _autodetect_device()
            self._ce = CrossEncoder(self._model_name, device=device)
        ce = self._ce

        def _score(pairs: list[tuple[str, str]]) -> list[float]:
            raw = ce.predict(pairs)  # type: ignore[union-attr]
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
