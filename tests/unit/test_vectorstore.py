"""Qdrant vector store wrapper. Uses qdrant-client's :memory: mode in unit tests."""

from __future__ import annotations

import pytest

from src.rag.vectorstore import QdrantVectorStore, VectorMatch
from src.types import Chunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, paper_id="p1", page_numbers=[1], text=text)


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
