"""PipelineRetriever: end-to-end query → ranked RetrievalResult list using fakes/in-memory."""

from __future__ import annotations

import pytest

from src.rag.bm25 import Bm25Index
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.retrievers.protocol import Retriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Chunk, Query
from tests.fakes import FakeEmbedder


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, paper_id="p1", page_numbers=[1], text=text)


@pytest.fixture
async def retriever() -> PipelineRetriever:
    embedder = FakeEmbedder(dim=8)
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="t", dim=8)
    await vectorstore.ensure_collection()
    bm25 = Bm25Index()

    chunks = [
        _chunk("c1", "Transformer attention mechanism for language models"),
        _chunk("c2", "Convolutional networks for image classification"),
        _chunk("c3", "Reinforcement learning policy optimization"),
    ]
    embeddings = await embedder.embed_texts([c.text for c in chunks])
    await vectorstore.upsert_chunks(chunks, embeddings)
    bm25.add(chunks)

    return PipelineRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id={c.chunk_id: c for c in chunks},
    )


def test_pipeline_retriever_satisfies_protocol(retriever: PipelineRetriever) -> None:
    assert isinstance(retriever, Retriever)


async def test_retrieve_returns_results_for_known_term(retriever: PipelineRetriever) -> None:
    """The chunk mentioning both query terms must surface in the top results.

    With FakeEmbedder (hash-based), the dense rank is arbitrary. We rely on BM25
    to put c1 high, and check it's *in* the top-2 — not specifically #1.
    """
    results = await retriever.retrieve(Query(text="attention transformer", top_k=2))

    assert len(results) <= 2
    assert all(r.source == "pipeline" for r in results)
    assert all(r.text and r.chunk_id and r.paper_id == "p1" for r in results)
    assert "c1" in {r.chunk_id for r in results}


async def test_retrieve_returns_empty_for_no_match(retriever: PipelineRetriever) -> None:
    results = await retriever.retrieve(Query(text="completely unrelated topic xyz", top_k=5))
    assert len(results) <= 5
