"""Synchronous CPU work on the request path must not stall the event loop.

The API runs one uvicorn worker on one Cloud Run instance, so a cross-encoder
rerank, a BM25 scan or a Docling parse executed inline freezes every other
request, /health included. Each test runs a deliberately slow stand-in for one
of those calls while a heartbeat task measures the longest gap between ticks.
"""

from __future__ import annotations

import asyncio
import gc
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

import pytest

from src.rag.bm25 import Bm25Hit, Bm25Index
from src.rag.rerank import BgeReranker
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Chunk, Paper, Query
from tests.fakes import FakeEmbedder

_BLOCK_S = 1.0
# Half of _BLOCK_S: far above Windows' ~15 ms timer granularity and a loaded
# test run's scheduling jitter, far below a blocked loop.
_MAX_STALL_S = 0.5


async def _longest_stall[T](awaitable: Awaitable[T]) -> tuple[T, float]:
    """Await `awaitable` while a heartbeat records the longest loop stall."""
    stall = 0.0
    done = asyncio.Event()
    started = asyncio.Event()

    async def heartbeat() -> None:
        nonlocal stall
        last = time.perf_counter()
        started.set()
        while not done.is_set():
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            stall = max(stall, now - last)
            last = now

    beat = asyncio.create_task(heartbeat())
    # The fakes never suspend, so without this the whole retrieve would run
    # before the heartbeat took its first timestamp and no stall could show.
    await started.wait()
    # Late in a full test run the heap is large enough that one gen-2 collection
    # stalls the loop for a few hundred ms, which is not what is measured here.
    gc.disable()
    try:
        result = await awaitable
    finally:
        gc.enable()
        done.set()
        await beat
    return result, stall


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
    def slow_scorer(pairs: list[tuple[str, str]]) -> list[float]:
        time.sleep(_BLOCK_S)
        return [0.5] * len(pairs)

    retriever = await _retriever(reranker=BgeReranker(scorer=slow_scorer))
    results, stall = await _longest_stall(retriever.retrieve(Query(text="alpha", top_k=2)))
    assert results
    assert stall < _MAX_STALL_S


async def test_bm25_search_runs_off_the_event_loop() -> None:
    class SlowBm25(Bm25Index):
        def search(
            self, query: str, top_k: int, *, paper_filter: str | None = None
        ) -> list[Bm25Hit]:
            time.sleep(_BLOCK_S)
            return super().search(query, top_k, paper_filter=paper_filter)

    retriever = await _retriever(bm25=SlowBm25())
    results, stall = await _longest_stall(retriever.retrieve(Query(text="alpha", top_k=2)))
    assert results
    assert stall < _MAX_STALL_S


async def test_docling_conversion_runs_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.ingestion import docling_chunker, docling_parser
    from src.ingestion.pipeline import ingest_paper

    def slow_convert(pdf_path: Path) -> docling_parser.DoclingConversion:
        time.sleep(_BLOCK_S)
        return docling_parser.DoclingConversion(document=object())

    monkeypatch.setattr(docling_parser, "convert_with_docling", slow_convert)
    monkeypatch.setattr(docling_chunker, "chunk_with_docling", lambda *a, **k: [])
    monkeypatch.setattr(docling_chunker, "paper_text_from_docling", lambda doc: "")

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF stub")
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="ing", dim=8)
    await vectorstore.ensure_collection()
    result, stall = await _longest_stall(
        ingest_paper(
            paper=Paper(paper_id="x", title="x", pdf_path=pdf),
            embedder=FakeEmbedder(dim=8),
            vectorstore=vectorstore,
            bm25=Bm25Index(),
        )
    )
    assert result.chunk_count == 0
    assert stall < _MAX_STALL_S
