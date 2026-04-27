"""BgeReranker: cross-encoder reranking with an injected scorer (model not downloaded in tests)."""

from __future__ import annotations

from src.rag.rerank import BgeReranker, RerankedHit
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
    fake_scores = {"low relevance text": 0.1, "extremely relevant text": 0.9, "medium relevance text": 0.5}
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
