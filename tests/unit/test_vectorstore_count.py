"""QdrantVectorStore.count — used by scripts/bootstrap_corpus.py for idempotency.

Uses the in-memory Qdrant client (`url=':memory:'`) so the test runs without
docker. Exercises the empty-collection, missing-collection, and populated
paths.
"""

from __future__ import annotations

import pytest

from src.rag.vectorstore import QdrantVectorStore
from src.types import Chunk


@pytest.mark.asyncio
async def test_count_returns_zero_when_collection_missing() -> None:
    """Newly-constructed store has no collection yet — count is 0, not an error."""
    store = QdrantVectorStore(url=":memory:", collection_name="not_yet_created", dim=4)
    assert await store.count() == 0


@pytest.mark.asyncio
async def test_count_returns_zero_when_collection_empty() -> None:
    """Created but empty collection counts as 0."""
    store = QdrantVectorStore(url=":memory:", collection_name="empty", dim=4)
    await store.ensure_collection()
    assert await store.count() == 0


@pytest.mark.asyncio
async def test_count_reflects_upserted_chunks() -> None:
    store = QdrantVectorStore(url=":memory:", collection_name="populated", dim=4)
    await store.ensure_collection()
    chunks = [
        Chunk(chunk_id=f"p1::p1::c{i}", paper_id="p1", page_numbers=[1], text=f"chunk {i}")
        for i in range(3)
    ]
    vectors = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.0, 0.1, 0.2]]
    await store.upsert_chunks(chunks, vectors)
    assert await store.count() == 3
