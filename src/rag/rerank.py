"""Cross-encoder reranker. Default: BGE reranker v2 m3 via sentence-transformers."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass

from src.observability.logging import get_logger
from src.types import Chunk

_log = get_logger(__name__)

ScorerFn = Callable[[list[tuple[str, str]]], list[float]]
"""A callable that scores (query, document) pairs into floats. Higher = more relevant."""

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

# Thresholds on a rerank score mean something only on the scale of the model
# they were measured with, so they are looked up by model id instead of being
# configured independently of the reranker. A model with no entry has no
# calibrated threshold: the refusal gate stays off and cascade routing runs
# both legs.
REFUSAL_THRESHOLDS: dict[str, float] = {
    # scripts/calibrate_refusal.py on golden v3 (ADR 0006, ADR 0009 follow-up).
    "BAAI/bge-reranker-v2-m3": 0.105,
}
CASCADE_THRESHOLDS: dict[str, float] = {
    # ADR 0010 verification value.
    "BAAI/bge-reranker-v2-m3": 0.85,
}

# ADR 0009 follow-up: caption-stub figure chunks (~50-150 chars of PDF caption
# text) and tiny table-only chunks empirically out-rank rich text chunks
# (target ~1200 chars) at the cross-encoder. Run ad4fab3bb28d / q11 demonstrated
# this: `p7::tab2` outranked `p6::c28` despite c28 carrying the answer. Length
# normalisation is a smooth penalty: 0 above the threshold, scales linearly to
# `length_penalty` at len=0, applied in the model's native score space. The 0.5
# default was tuned on bge-reranker-v2-m3, which sentence-transformers runs
# through a sigmoid (one output label, no activation in its config), so it is
# 0.5 off a [0, 1] probability. On a logit model such as ms-marco-MiniLM the
# same 0.5 is a much milder nudge. q8's "8 tasks and 65 instances" (~250 chars)
# sits above the threshold and is untouched.
_DEFAULT_LENGTH_THRESHOLD = 300
_DEFAULT_LENGTH_PENALTY = 0.5


def _length_penalty_for(text_len: int, threshold: int, penalty_max: float) -> float:
    """Linear penalty: 0 at threshold (and above), `penalty_max` at len=0.

    Smoothly punishes short docs at the cross-encoder layer: caption-stub
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
    we don't want to load when an injected scorer is in use (in tests, say).
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
        scorer_returns_logits: bool = False,
    ) -> None:
        self._injected_scorer = scorer
        # Whether native scores are raw logits (mapped through a sigmoid before
        # they leave rerank()). Read off the loaded model's activation;
        # `scorer_returns_logits` covers an injected scorer.
        self._outputs_logits = scorer_returns_logits
        self._model_name = model_name
        self._device = device
        self._ce: object | None = None
        # rerank() runs in worker threads; concurrent first calls must not each
        # load the model.
        self._load_lock = threading.Lock()
        self._length_norm = length_norm
        self._length_threshold = length_threshold
        self._length_penalty = length_penalty

    def _resolve_scorer(self) -> ScorerFn:
        if self._injected_scorer is not None:
            return self._injected_scorer
        with self._load_lock:
            if self._ce is None:
                from sentence_transformers import CrossEncoder

                device = self._device if self._device is not None else _autodetect_device()
                self._ce = CrossEncoder(self._model_name, device=device)
                activation = getattr(self._ce, "activation_fn", None)
                self._outputs_logits = type(activation).__name__ == "Identity"
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

        Ranking uses the native (penalised) score. The returned `rerank_score`
        is probability-scaled for every model: a logit model's penalised score
        goes through a sigmoid (order unchanged), a sigmoid model's is returned
        as-is and can dip below zero by up to the length penalty.
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
            RerankedHit(
                chunk_id=chunk.chunk_id,
                rerank_score=_sigmoid(score) if self._outputs_logits else score,
            )
            for chunk, score in ranked[:top_k]
        ]


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def calibrated_refusal_threshold(model: str | None) -> float | None:
    return REFUSAL_THRESHOLDS.get(model) if model else None


def calibrated_cascade_threshold(model: str | None) -> float | None:
    return CASCADE_THRESHOLDS.get(model) if model else None
