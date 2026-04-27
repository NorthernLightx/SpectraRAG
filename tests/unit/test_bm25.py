"""BM25 in-process index over Chunk text."""

from __future__ import annotations

from src.rag.bm25 import Bm25Index
from src.types import Chunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, paper_id="p1", page_numbers=[1], text=text)


def test_search_returns_chunks_ranked_by_term_overlap() -> None:
    index = Bm25Index()
    index.add(
        [
            _chunk("c1", "Attention is all you need transformer"),
            _chunk("c2", "Convolutional neural networks for vision"),
            _chunk("c3", "Recurrent networks and attention mechanisms"),
        ]
    )
    hits = index.search("attention transformer", top_k=2)

    assert len(hits) == 2
    chunk_ids = [hit.chunk_id for hit in hits]
    assert chunk_ids[0] == "c1"
    assert hits[0].score >= hits[1].score


def test_search_returns_empty_when_index_empty() -> None:
    index = Bm25Index()
    assert index.search("anything", top_k=5) == []


def test_search_caps_to_top_k() -> None:
    index = Bm25Index()
    index.add([_chunk(f"c{i}", "common token") for i in range(10)])
    assert len(index.search("common", top_k=3)) == 3


def test_add_after_search_rebuilds_index() -> None:
    index = Bm25Index()
    index.add([_chunk("c1", "alpha")])
    index.search("alpha", top_k=1)
    index.add([_chunk("c2", "alpha beta")])
    hits = index.search("beta", top_k=1)
    assert hits[0].chunk_id == "c2"
