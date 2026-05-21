"""QdrantVectorStore.scroll_chunks — payload round-trip used to seed the API
retriever at startup. ``_wire_retriever_from_settings`` reads every chunk back
from Qdrant to rebuild BM25 + chunks_by_id without a separate manifest file,
so this test covers fidelity (every Chunk field survives) plus the empty /
missing-collection edge cases the wire path handles.
"""

from __future__ import annotations

import pytest

from src.rag.vectorstore import QdrantVectorStore
from src.types import Chunk


def _vec() -> list[float]:
    return [0.1, 0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_scroll_returns_empty_when_collection_missing() -> None:
    store = QdrantVectorStore(url=":memory:", collection_name="never_created", dim=4)
    assert await store.scroll_chunks() == []


@pytest.mark.asyncio
async def test_scroll_returns_empty_for_empty_collection() -> None:
    store = QdrantVectorStore(url=":memory:", collection_name="empty", dim=4)
    await store.ensure_collection()
    assert await store.scroll_chunks() == []


@pytest.mark.asyncio
async def test_scroll_round_trips_full_chunk_fields() -> None:
    """Section, context, page_numbers, metadata all survive upsert + scroll."""
    store = QdrantVectorStore(url=":memory:", collection_name="round_trip", dim=4)
    await store.ensure_collection()
    chunks = [
        Chunk(
            chunk_id="paper-x::p3::c0",
            paper_id="paper-x",
            page_numbers=[3, 4],
            text="dense paragraph text",
            section="Introduction",
            context="This paper studies...",
            metadata={"source": "arxiv"},
        ),
        Chunk(
            chunk_id="paper-y::p1::c0",
            paper_id="paper-y",
            page_numbers=[1],
            text="another chunk",
            section=None,
            context=None,
            metadata={},
        ),
    ]
    await store.upsert_chunks(chunks, [_vec(), _vec()])

    scrolled = await store.scroll_chunks()

    by_id = {c.chunk_id: c for c in scrolled}
    assert set(by_id) == {"paper-x::p3::c0", "paper-y::p1::c0"}
    assert by_id["paper-x::p3::c0"].section == "Introduction"
    assert by_id["paper-x::p3::c0"].context == "This paper studies..."
    assert by_id["paper-x::p3::c0"].page_numbers == [3, 4]
    assert by_id["paper-x::p3::c0"].metadata == {"source": "arxiv"}
    assert by_id["paper-y::p1::c0"].section is None
    assert by_id["paper-y::p1::c0"].context is None


@pytest.mark.asyncio
async def test_scroll_paginates_past_default_batch() -> None:
    """A collection larger than the scroll batch still returns every chunk —
    proves the pagination loop terminates correctly on a non-trivial size."""
    store = QdrantVectorStore(url=":memory:", collection_name="paged", dim=4)
    await store.ensure_collection()
    n = 50
    chunks = [
        Chunk(chunk_id=f"p::p1::c{i}", paper_id="p", page_numbers=[1], text=f"chunk {i}")
        for i in range(n)
    ]
    await store.upsert_chunks(chunks, [_vec() for _ in range(n)])

    # Force pagination by using a small batch.
    scrolled = await store.scroll_chunks(batch_size=8)

    assert len(scrolled) == n
    assert {c.chunk_id for c in scrolled} == {f"p::p1::c{i}" for i in range(n)}


@pytest.mark.asyncio
async def test_delete_collection_clears_points() -> None:
    """`--force` re-ingest relies on this: a real drop, so a re-ingest can't
    leave stale chunks behind (the contamination the bootstrap --force hit when
    it only called create-if-absent ensure_collection)."""
    store = QdrantVectorStore(url=":memory:", collection_name="to_drop", dim=4)
    await store.ensure_collection()
    await store.upsert_chunks(
        [Chunk(chunk_id="p::p1::c0", paper_id="p", page_numbers=[1], text="x")], [_vec()]
    )
    assert await store.count() == 1

    await store.delete_collection()

    assert await store.count() == 0
    assert await store.scroll_chunks() == []


@pytest.mark.asyncio
async def test_delete_collection_missing_is_noop() -> None:
    store = QdrantVectorStore(url=":memory:", collection_name="never_made", dim=4)
    await store.delete_collection()  # must not raise
    assert await store.count() == 0
