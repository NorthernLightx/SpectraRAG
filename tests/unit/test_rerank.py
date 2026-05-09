"""BgeReranker: cross-encoder reranking with an injected scorer (model not downloaded in tests)."""

from __future__ import annotations

import pytest

from src.rag.rerank import BgeReranker, RerankedHit, _length_penalty_for
from src.types import Chunk


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, paper_id="p1", page_numbers=[1], text=text)


def test_rerank_orders_by_scorer_output() -> None:
    """Inject a scorer that returns deterministic scores; verify ordering."""
    chunks = [
        _chunk("c1", "low relevance text"),
        _chunk("c2", "extremely relevant text"),
        _chunk("c3", "medium relevance text"),
    ]
    fake_scores = {
        "low relevance text": 0.1,
        "extremely relevant text": 0.9,
        "medium relevance text": 0.5,
    }
    reranker = BgeReranker(scorer=lambda pairs: [fake_scores[doc] for _, doc in pairs])

    hits = reranker.rerank("any query", chunks, top_k=3)

    assert [h.chunk_id for h in hits] == ["c2", "c3", "c1"]
    assert hits[0].rerank_score == 0.9
    assert all(isinstance(h, RerankedHit) for h in hits)


def test_rerank_caps_to_top_k() -> None:
    chunks = [_chunk(f"c{i}", f"text {i}") for i in range(10)]
    reranker = BgeReranker(scorer=lambda pairs: [float(i) for i in range(len(pairs))])
    hits = reranker.rerank("q", chunks, top_k=3)
    assert len(hits) == 3


def test_rerank_handles_empty() -> None:
    reranker = BgeReranker(scorer=lambda _: [])
    assert reranker.rerank("q", [], top_k=5) == []


def test_rerank_invokes_scorer_with_query_doc_pairs() -> None:
    captured: list[list[tuple[str, str]]] = []

    def scorer(pairs: list[tuple[str, str]]) -> list[float]:
        captured.append(pairs)
        return [0.0] * len(pairs)

    reranker = BgeReranker(scorer=scorer)
    chunks = [_chunk("c1", "doc one"), _chunk("c2", "doc two")]
    reranker.rerank("my query", chunks, top_k=2)

    assert captured == [[("my query", "doc one"), ("my query", "doc two")]]


# ADR 0009 follow-up: length-norm penalises caption-stub chunks at rerank.


def test_length_penalty_is_zero_above_threshold() -> None:
    assert _length_penalty_for(text_len=300, threshold=300, penalty_max=0.5) == 0.0
    assert _length_penalty_for(text_len=1200, threshold=300, penalty_max=0.5) == 0.0


def test_length_penalty_is_max_at_zero_length() -> None:
    assert _length_penalty_for(text_len=0, threshold=300, penalty_max=0.5) == 0.5
    # Negative or otherwise pathological input clamps to penalty_max.
    assert _length_penalty_for(text_len=-5, threshold=300, penalty_max=0.5) == 0.5


def test_length_penalty_scales_linearly_below_threshold() -> None:
    # At half the threshold, penalty is half of penalty_max.
    assert _length_penalty_for(text_len=150, threshold=300, penalty_max=0.5) == pytest.approx(0.25)
    # At a quarter, penalty is 3/4 of penalty_max.
    assert _length_penalty_for(text_len=75, threshold=300, penalty_max=0.5) == pytest.approx(0.375)


def test_length_norm_off_by_default() -> None:
    """Existing behavior preserved when the flag isn't set."""
    chunks = [_chunk("short", "x"), _chunk("long", "x" * 1000)]
    # Same raw score; without length-norm, the order is stable (short first per
    # input order since both score equal — sorted is stable).
    reranker = BgeReranker(scorer=lambda pairs: [1.0, 1.0])
    hits = reranker.rerank("q", chunks, top_k=2)
    assert [h.rerank_score for h in hits] == [1.0, 1.0]


def test_length_norm_penalises_short_chunks() -> None:
    """Short stub at higher raw score loses to long chunk at slightly lower score."""
    short = _chunk("stub", "Figure 1: An overview.")  # 22 chars
    long = _chunk("text", "x" * 1200)
    # Short raw=1.0, long raw=0.7. Without length-norm, short wins.
    reranker = BgeReranker(
        scorer=lambda pairs: [1.0, 0.7],
        length_norm=True,
        length_threshold=300,
        length_penalty=0.5,
    )
    hits = reranker.rerank("q", [short, long], top_k=2)
    # short adjusted: 1.0 - 0.5 * (1 - 22/300) = 1.0 - 0.4633 = 0.5367
    # long adjusted: 0.7 (above threshold, no penalty)
    # long wins.
    assert [h.chunk_id for h in hits] == ["text", "stub"]
    assert hits[0].rerank_score == pytest.approx(0.7)
    assert hits[1].rerank_score == pytest.approx(0.5367, abs=1e-3)


def test_length_norm_does_not_destroy_short_clear_winner() -> None:
    """A short doc with overwhelming raw score still beats a long doc — penalty
    is gentle by design (default 0.5 on a [-5, 5] logit range)."""
    short = _chunk("short", "8 tasks and 65 instances.")  # 25 chars, q8-like
    long = _chunk("long", "x" * 1200)
    reranker = BgeReranker(
        scorer=lambda pairs: [3.0, 0.5],
        length_norm=True,
        length_threshold=300,
        length_penalty=0.5,
    )
    hits = reranker.rerank("q", [short, long], top_k=2)
    # short adjusted: 3.0 - 0.5 * (1 - 25/300) = 3.0 - 0.458 = 2.542
    # long: 0.5
    # short still wins by 2 points.
    assert hits[0].chunk_id == "short"


def test_length_norm_zero_penalty_is_no_op() -> None:
    """Setting length_penalty=0 with length_norm=True is identical to off."""
    chunks = [_chunk("short", "x"), _chunk("long", "x" * 1000)]
    reranker = BgeReranker(
        scorer=lambda pairs: [1.0, 0.5],
        length_norm=True,
        length_penalty=0.0,
    )
    hits = reranker.rerank("q", chunks, top_k=2)
    assert [h.chunk_id for h in hits] == ["short", "long"]


def test_length_norm_handles_empty_text_chunks() -> None:
    """A chunk with empty text gets the full penalty, not a divide-by-zero."""
    empty = _chunk("e", "")
    long = _chunk("l", "x" * 1000)
    reranker = BgeReranker(
        scorer=lambda pairs: [1.0, 0.7],
        length_norm=True,
        length_threshold=300,
        length_penalty=0.5,
    )
    hits = reranker.rerank("q", [empty, long], top_k=2)
    assert hits[0].chunk_id == "l"
    # empty's score is 1.0 - 0.5 = 0.5; long's is 0.7
    assert hits[0].rerank_score == pytest.approx(0.7)
    assert hits[1].rerank_score == pytest.approx(0.5)
