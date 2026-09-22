"""Synchronous CPU work on the request path must not run on the event loop.

The API runs one uvicorn worker on one Cloud Run instance, so a cross-encoder
rerank, a BM25 scan or a Docling parse executed inline freezes every other
request, /health included. Each test records the thread a stand-in for one of
those calls ran on and checks it is not the event loop's thread.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from src.rag.bm25 import Bm25Hit, Bm25Index
from src.rag.rerank import BgeReranker
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Chunk, Paper, Query
from tests.fakes import FakeEmbedder


async def _retriever(**kwargs: Any) -> PipelineRetriever:
    embedder = FakeEmbedder(dim=8)
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="offload", dim=8)
    await vectorstore.ensure_collection()
    chunks = [
        Chunk(chunk_id="p1::p1::c0", paper_id="p1", page_numbers=[1], text="alpha beta"),
        Chunk(chunk_id="p1::p1::c1", paper_id="p1", page_numbers=[1], text="gamma delta"),
    ]
    await vectorstore.upsert_chunks(chunks, await embedder.embed_texts([c.text for c in chunks]))
    bm25 = kwargs.pop("bm25", None) or Bm25Index()
    bm25.add(chunks)
    return PipelineRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id={c.chunk_id: c for c in chunks},
        **kwargs,
    )


async def test_rerank_runs_off_the_event_loop() -> None:
    ran_on: list[int] = []

    def scorer(pairs: list[tuple[str, str]]) -> list[float]:
        ran_on.append(threading.get_ident())
        return [0.5] * len(pairs)

    retriever = await _retriever(reranker=BgeReranker(scorer=scorer))
    assert await retriever.retrieve(Query(text="alpha", top_k=2))
    assert ran_on and ran_on[0] != threading.get_ident()


async def test_bm25_search_runs_off_the_event_loop() -> None:
    ran_on: list[int] = []

    class RecordingBm25(Bm25Index):
        def search(
            self, query: str, top_k: int, *, paper_filter: str | None = None
        ) -> list[Bm25Hit]:
            ran_on.append(threading.get_ident())
            return super().search(query, top_k, paper_filter=paper_filter)

    retriever = await _retriever(bm25=RecordingBm25())
    assert await retriever.retrieve(Query(text="alpha", top_k=2))
    assert ran_on and ran_on[0] != threading.get_ident()


async def test_docling_conversion_runs_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ingestion import docling_chunker, docling_parser
    from src.ingestion.pipeline import ingest_paper

    ran_on: list[int] = []

    def convert(pdf_path: Path) -> docling_parser.DoclingConversion:
        ran_on.append(threading.get_ident())
        return docling_parser.DoclingConversion(document=object())

    monkeypatch.setattr(docling_parser, "convert_with_docling", convert)
    monkeypatch.setattr(docling_chunker, "chunk_with_docling", lambda *a, **k: [])
    monkeypatch.setattr(docling_chunker, "paper_text_from_docling", lambda doc: "")

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF stub")
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="ing", dim=8)
    await vectorstore.ensure_collection()
    result = await ingest_paper(
        paper=Paper(paper_id="x", title="x", pdf_path=pdf),
        embedder=FakeEmbedder(dim=8),
        vectorstore=vectorstore,
        bm25=Bm25Index(),
    )
    assert result.chunk_count == 0
    assert ran_on and ran_on[0] != threading.get_ident()
