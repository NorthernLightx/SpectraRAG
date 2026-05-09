"""Cross-encoder reranker. Default: BGE reranker v2 m3 via sentence-transformers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.observability.logging import get_logger
from src.types import Chunk

_log = get_logger(__name__)

ScorerFn = Callable[[list[tuple[str, str]]], list[float]]
"""A callable that scores (query, document) pairs into floats. Higher = more relevant."""

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

# ADR 0009 follow-up: caption-stub figure chunks (~50-150 chars of PDF caption
# text) and tiny table-only chunks empirically out-rank rich text chunks
# (target ~1200 chars) at the cross-encoder. Run ad4fab3bb28d / q11 demonstrated
# this — `p7::tab2` outranked `p6::c28` despite c28 carrying the answer. Length
# normalisation is a smooth penalty: 0 above the threshold, scales linearly to
# `length_penalty` at len=0. Defaults are calibrated for bge-reranker-v2-m3
# scores (typically [-5, 5] logits): a 0.5 penalty is enough to displace a
# borderline stub but not destroy a legitimately short answer (e.g. q8's
# "8 tasks and 65 instances" — ~250 chars, above threshold so untouched).
_DEFAULT_LENGTH_THRESHOLD = 300
_DEFAULT_LENGTH_PENALTY = 0.5


def _length_penalty_for(text_len: int, threshold: int, penalty_max: float) -> float:
    """Linear penalty: 0 at threshold (and above), `penalty_max` at len=0.

    Smoothly punishes short docs at the cross-encoder layer — caption-stub
    figure chunks (~80 chars) get nearly the full penalty; legitimately short
    factual answers (~250 chars) get a small fraction; full text chunks
    (>= threshold) are untouched. ADR 0009 §"What this leaves open" #1.
    """
    if text_len <= 0:
        return penalty_max
    if text_len >= threshold:
        return 0.0
    return penalty_max * (1.0 - text_len / threshold)


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
        length_norm: bool = False,
        length_threshold: int = _DEFAULT_LENGTH_THRESHOLD,
        length_penalty: float = _DEFAULT_LENGTH_PENALTY,
    ) -> None:
        self._injected_scorer = scorer
        self._model_name = model_name
        self._device = device
        self._ce: object | None = None
        self._length_norm = length_norm
        self._length_threshold = length_threshold
        self._length_penalty = length_penalty

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
        """Score (query, chunk.text) pairs, sort descending, cap to top_k.

        When `length_norm=True`, subtracts a smooth length penalty from each
        chunk's raw score before sorting. The penalty is calibrated to displace
        caption-stub chunks but leave legitimately short answers untouched.
        See ADR 0009 §"What this leaves open" for the empirical motivation
        and `_length_penalty_for` for the formula.
        """
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        raw_scores = self._resolve_scorer()(pairs)
        if self._length_norm:
            penalised: list[float] = []
            for c, raw in zip(candidates, raw_scores, strict=True):
                penalty = _length_penalty_for(
                    len(c.text), self._length_threshold, self._length_penalty
                )
                penalised.append(float(raw) - penalty)
            scores: list[float] = penalised
        else:
            scores = [float(s) for s in raw_scores]
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [
            RerankedHit(chunk_id=chunk.chunk_id, rerank_score=score)
            for chunk, score in ranked[:top_k]
        ]
