"""Qdrant vector store wrapper. Uses qdrant-client's :memory: mode in unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.vectorstore import QdrantVectorStore, VectorMatch
from src.types import Chunk


def _chunk(chunk_id: str, text: str, paper_id: str = "p1") -> Chunk:
    return Chunk(chunk_id=chunk_id, paper_id=paper_id, page_numbers=[1], text=text)


@pytest.fixture
def store() -> QdrantVectorStore:
    return QdrantVectorStore(url=":memory:", collection_name="test", dim=4)


async def test_ensure_collection_idempotent(store: QdrantVectorStore) -> None:
    await store.ensure_collection()
    await store.ensure_collection()


async def test_upsert_and_search(store: QdrantVectorStore) -> None:
    await store.ensure_collection()
    await store.upsert_chunks(
        [_chunk("c1", "x"), _chunk("c2", "y")],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )

    hits = await store.search([1.0, 0.0, 0.0, 0.0], top_k=2)

    assert len(hits) == 2
    assert all(isinstance(hit, VectorMatch) for hit in hits)
    assert hits[0].chunk_id == "c1"
    assert hits[0].score >= hits[1].score


async def test_upsert_mismatched_lengths_raises(store: QdrantVectorStore) -> None:
    await store.ensure_collection()
    with pytest.raises(ValueError, match="length mismatch"):
        await store.upsert_chunks([_chunk("c1", "x")], [[1.0, 0.0], [0.0, 1.0]])


async def test_search_empty_collection_returns_empty(store: QdrantVectorStore) -> None:
    await store.ensure_collection()
    assert await store.search([0.0, 0.0, 0.0, 0.0], top_k=5) == []


# ADR 0009 follow-up: paper_filter scopes Qdrant search to a single paper.


async def test_search_with_paper_filter_drops_other_papers(store: QdrantVectorStore) -> None:
    await store.ensure_collection()
    await store.upsert_chunks(
        [
            _chunk("paperA::c0", "x", paper_id="paperA"),
            _chunk("paperB::c0", "y", paper_id="paperB"),
        ],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    )
    hits = await store.search([1.0, 0.0, 0.0, 0.0], top_k=10, paper_filter="paperB")
    assert [h.chunk_id for h in hits] == ["paperB::c0"]


async def test_search_no_filter_unchanged(store: QdrantVectorStore) -> None:
    """Default path (no filter) returns all matching points."""
    await store.ensure_collection()
    await store.upsert_chunks(
        [
            _chunk("paperA::c0", "x", paper_id="paperA"),
            _chunk("paperB::c0", "y", paper_id="paperB"),
        ],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    )
    hits = await store.search([1.0, 0.0, 0.0, 0.0], top_k=10)
    assert {h.chunk_id for h in hits} == {"paperA::c0", "paperB::c0"}


async def test_search_with_unmatched_paper_filter_returns_empty(store: QdrantVectorStore) -> None:
    await store.ensure_collection()
    await store.upsert_chunks(
        [_chunk("paperA::c0", "x", paper_id="paperA")],
        [[1.0, 0.0, 0.0, 0.0]],
    )
    assert await store.search([1.0, 0.0, 0.0, 0.0], top_k=10, paper_filter="paperZ") == []


async def test_path_mode_persists_across_clients(tmp_path: Path) -> None:
    """`url='path:<dir>'` writes a sqlite-backed local store. The deploy bakes
    a snapshot into the Docker image with this mode so there's no external
    Qdrant service. Verifies a chunk written by one client is readable by a
    second client opened against the same path."""
    url = f"path:{tmp_path / 'q'}"
    store = QdrantVectorStore(url=url, collection_name="round", dim=4)
    await store.ensure_collection()
    chunks = [Chunk(chunk_id="paper::p1::c0", paper_id="paper", page_numbers=[1], text="hi")]
    await store.upsert_chunks(chunks, [[0.1, 0.2, 0.3, 0.4]])
    assert await store.count() == 1
    # Release the file lock before reopening — qdrant-client local mode uses
    # portalocker to enforce single-writer; the deploy never opens twice but
    # this test does.
    await store._client.close()

    reopened = QdrantVectorStore(url=url, collection_name="round", dim=4)
    assert await reopened.count() == 1
    scrolled = await reopened.scroll_chunks()
    assert [c.chunk_id for c in scrolled] == ["paper::p1::c0"]
    await reopened._client.close()
