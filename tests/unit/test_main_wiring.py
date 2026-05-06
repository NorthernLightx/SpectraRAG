"""_wire_retriever_from_settings — startup wiring of the production retriever.

Covers the three paths the lifespan handler walks at API startup: (1) wires a
PipelineRetriever when Qdrant has chunks, (2) silently skips when the
collection is empty, (3) silently skips when Qdrant is unreachable. The wire
function is the only thing that closes the gap between bootstrap_corpus.py
and a live /answer — without it, /answer returns 503 even after a successful
ingest.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from src.api.deps import _RetrieverState
from src.api.main import _wire_retriever_from_settings
from src.config.settings import Settings
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Chunk
from tests.fakes import FakeEmbedder


@pytest.fixture(autouse=True)
def _reset_retriever_state() -> Iterator[None]:
    """Module-level retriever leaks across tests; reset around each."""
    _RetrieverState.instance = None
    yield
    _RetrieverState.instance = None


def _settings(*, qdrant_url: str = ":memory:") -> Settings:
    return Settings(
        env="test",
        log_level="INFO",
        qdrant_url=qdrant_url,
        corpus_collection="wiring_test",
        rerank_top_k=20,
    )


def _vec(dim: int) -> list[float]:
    return [0.1] * dim


@pytest.mark.asyncio
async def test_wire_returns_false_when_collection_empty() -> None:
    """No corpus ingested → wire is a no-op, /answer falls through to 503."""
    embedder = FakeEmbedder(dim=8)
    store = QdrantVectorStore(url=":memory:", collection_name="wiring_test", dim=embedder.dim)
    await store.ensure_collection()  # exists but no points

    wired = await _wire_retriever_from_settings(_settings(), embedder=embedder, vectorstore=store)

    assert wired is False
    assert _RetrieverState.instance is None


@pytest.mark.asyncio
async def test_wire_returns_false_when_collection_does_not_exist() -> None:
    """Even more degenerate: collection was never created. Same outcome."""
    embedder = FakeEmbedder(dim=8)
    store = QdrantVectorStore(url=":memory:", collection_name="never_made", dim=embedder.dim)

    wired = await _wire_retriever_from_settings(_settings(), embedder=embedder, vectorstore=store)

    assert wired is False
    assert _RetrieverState.instance is None


@pytest.mark.asyncio
async def test_wire_returns_false_on_qdrant_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unreachable Qdrant raises during scroll → caught, logged, retriever stays None."""

    class _BoomStore:
        async def scroll_chunks(self) -> list[Chunk]:
            raise ConnectionError("Qdrant unreachable")

    embedder = FakeEmbedder(dim=8)

    wired = await _wire_retriever_from_settings(
        _settings(qdrant_url="http://nowhere.invalid:6333"),
        embedder=embedder,
        vectorstore=_BoomStore(),  # type: ignore[arg-type]
    )

    assert wired is False
    assert _RetrieverState.instance is None


@pytest.mark.asyncio
async def test_wire_populates_pipeline_retriever_from_qdrant() -> None:
    """The full happy path: chunks live in Qdrant → wire builds BM25 +
    chunks_by_id and registers a PipelineRetriever via set_retriever()."""
    embedder = FakeEmbedder(dim=8)
    store = QdrantVectorStore(url=":memory:", collection_name="wiring_test", dim=embedder.dim)
    await store.ensure_collection()
    chunks = [
        Chunk(
            chunk_id=f"paper::p1::c{i}",
            paper_id="paper",
            page_numbers=[1],
            text=f"chunk text {i}",
            section="Intro" if i == 0 else None,
        )
        for i in range(3)
    ]
    await store.upsert_chunks(chunks, [_vec(embedder.dim) for _ in chunks])

    wired = await _wire_retriever_from_settings(_settings(), embedder=embedder, vectorstore=store)

    assert wired is True
    assert isinstance(_RetrieverState.instance, PipelineRetriever)


@pytest.mark.asyncio
async def test_wired_retriever_can_serve_a_query() -> None:
    """Sanity check beyond construction: the wired retriever actually returns
    chunks (so set_retriever didn't silently register a broken instance)."""
    from src.types import Query

    embedder = FakeEmbedder(dim=8)
    store = QdrantVectorStore(url=":memory:", collection_name="wiring_test", dim=embedder.dim)
    await store.ensure_collection()
    chunks = [
        Chunk(
            chunk_id="paper::p1::c0",
            paper_id="paper",
            page_numbers=[1],
            text="multi-modal retrieval over papers",
        ),
        Chunk(
            chunk_id="paper::p1::c1",
            paper_id="paper",
            page_numbers=[1],
            text="some unrelated content",
        ),
    ]
    await store.upsert_chunks(chunks, [_vec(embedder.dim) for _ in chunks])

    wired = await _wire_retriever_from_settings(_settings(), embedder=embedder, vectorstore=store)
    assert wired is True

    retriever = _RetrieverState.instance
    assert retriever is not None
    results = await retriever.retrieve(Query(text="multi-modal retrieval", top_k=2))
    assert len(results) >= 1
    assert {r.chunk_id for r in results}.issubset({c.chunk_id for c in chunks})
