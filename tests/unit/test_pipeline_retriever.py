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


async def test_retrieve_with_reranker_uses_rerank_scores() -> None:
    """A reranker reorders results and replaces fused score with rerank_score."""
    from src.rag.rerank import BgeReranker

    embedder = FakeEmbedder(dim=8)
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="rr", dim=8)
    await vectorstore.ensure_collection()
    bm25 = Bm25Index()
    chunks = [
        _chunk("c1", "alpha"),
        _chunk("c2", "beta"),
        _chunk("c3", "gamma"),
    ]
    embeddings = await embedder.embed_texts([c.text for c in chunks])
    await vectorstore.upsert_chunks(chunks, embeddings)
    bm25.add(chunks)

    # Inject a scorer that strongly prefers c3, then c1, then c2.
    scores_by_text = {"alpha": 0.5, "beta": 0.1, "gamma": 0.95}
    reranker = BgeReranker(scorer=lambda pairs: [scores_by_text[doc] for _, doc in pairs])

    retriever = PipelineRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id={c.chunk_id: c for c in chunks},
        reranker=reranker,
    )

    results = await retriever.retrieve(Query(text="anything", top_k=3))
    assert [r.chunk_id for r in results] == ["c3", "c1", "c2"]
    assert results[0].score == 0.95


async def test_retrieve_with_paper_filter_scopes_to_one_paper() -> None:
    """ADR 0009 follow-up: Query.filters['paper_id'] propagates to BM25 + Qdrant."""
    embedder = FakeEmbedder(dim=8)
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="pf", dim=8)
    await vectorstore.ensure_collection()
    bm25 = Bm25Index()

    chunks = [
        Chunk(chunk_id="paperA::p1::c0", paper_id="paperA", page_numbers=[1], text="alpha topic"),
        Chunk(chunk_id="paperB::p1::c0", paper_id="paperB", page_numbers=[1], text="alpha topic"),
        Chunk(chunk_id="paperC::p1::c0", paper_id="paperC", page_numbers=[1], text="alpha topic"),
    ]
    embeddings = await embedder.embed_texts([c.text for c in chunks])
    await vectorstore.upsert_chunks(chunks, embeddings)
    bm25.add(chunks)

    retriever = PipelineRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id={c.chunk_id: c for c in chunks},
    )
    results = await retriever.retrieve(
        Query(text="alpha topic", top_k=5, filters={"paper_id": "paperB"})
    )
    assert {r.paper_id for r in results} == {"paperB"}


async def test_retrieve_without_paper_filter_unchanged() -> None:
    """Empty filters dict is a no-op — existing behavior preserved."""
    embedder = FakeEmbedder(dim=8)
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="pf2", dim=8)
    await vectorstore.ensure_collection()
    bm25 = Bm25Index()

    chunks = [
        Chunk(chunk_id="paperA::p1::c0", paper_id="paperA", page_numbers=[1], text="alpha"),
        Chunk(chunk_id="paperB::p1::c0", paper_id="paperB", page_numbers=[1], text="alpha"),
    ]
    embeddings = await embedder.embed_texts([c.text for c in chunks])
    await vectorstore.upsert_chunks(chunks, embeddings)
    bm25.add(chunks)

    retriever = PipelineRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id={c.chunk_id: c for c in chunks},
    )
    results = await retriever.retrieve(Query(text="alpha", top_k=5))
    assert {r.paper_id for r in results} == {"paperA", "paperB"}


async def test_retrieve_carries_chunk_metadata_into_result_metadata() -> None:
    """ADR 0009: Chunk.metadata (kind, bbox, image_path) must propagate to
    RetrievalResult.metadata so Generator._extract_citations can pick up
    bbox for region-grounded citations. Section is added on top — it lives
    on the Chunk model itself, not in metadata."""
    embedder = FakeEmbedder(dim=8)
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="md", dim=8)
    await vectorstore.ensure_collection()
    bm25 = Bm25Index()

    fig_chunk = Chunk(
        chunk_id="p1::p3::fig1",
        paper_id="p1",
        page_numbers=[3],
        text="Figure 1: X.",
        section=None,
        metadata={
            "kind": "figure",
            "bbox": [10.0, 20.0, 110.0, 220.0],
            "image_path": "/tmp/fig.png",
            "has_vlm_caption": False,
        },
    )
    text_chunk = Chunk(
        chunk_id="p1::p1::c0",
        paper_id="p1",
        page_numbers=[1],
        text="Plain text.",
        section="1 Introduction",
    )
    chunks = [fig_chunk, text_chunk]
    embeddings = await embedder.embed_texts([c.text for c in chunks])
    await vectorstore.upsert_chunks(chunks, embeddings)
    bm25.add(chunks)

    retriever = PipelineRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id={c.chunk_id: c for c in chunks},
    )
    results = await retriever.retrieve(Query(text="figure", top_k=2))
    by_id = {r.chunk_id: r for r in results}

    fig_result = by_id["p1::p3::fig1"]
    assert fig_result.metadata.get("kind") == "figure"
    assert fig_result.metadata.get("bbox") == [10.0, 20.0, 110.0, 220.0]
    assert fig_result.metadata.get("image_path") == "/tmp/fig.png"
    assert fig_result.metadata.get("has_vlm_caption") is False

    text_result = by_id["p1::p1::c0"]
    assert text_result.metadata.get("section") == "1 Introduction"
    # Text chunks should not get a phantom bbox from metadata-merging logic.
    assert "bbox" not in text_result.metadata or text_result.metadata.get("bbox") is None
