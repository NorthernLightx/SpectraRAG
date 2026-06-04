"""Ingestion pipeline: paper → pages → chunks → indexed in BM25 + vectorstore."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz
import pytest

from src.ingestion.pipeline import IngestedPaper, ingest_paper
from src.llm.protocol import ChatResponse, Message
from src.rag.bm25 import Bm25Index
from src.rag.vectorstore import QdrantVectorStore
from src.types import Paper
from tests.fakes import FakeEmbedder

# Real docling conversion downloads/loads a HF figure-classifier model — too
# heavy for the fast pre-push gauntlet. Runs in CI, skipped by the local hook.
pytestmark = pytest.mark.slow


class _StubLLM:
    def __init__(self, text: str = "blurb") -> None:
        self.text = text
        self.calls = 0

    async def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.calls += 1
        return ChatResponse(text=self.text, model=model, tokens_in=1, tokens_out=1)


def _make_pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1 Introduction\n\nThis paper studies attention.", fontsize=11)
    pdf_path = tmp_path / "tiny.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


async def test_ingest_paper_indexes_chunks_in_both_stores(tmp_path: Path) -> None:
    pdf_path = _make_pdf(tmp_path)
    paper = Paper(paper_id="p1", title="Attention paper", pdf_path=pdf_path)
    embedder = FakeEmbedder(dim=8)
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="t", dim=8)
    await vectorstore.ensure_collection()
    bm25 = Bm25Index()

    result = await ingest_paper(paper=paper, embedder=embedder, vectorstore=vectorstore, bm25=bm25)

    assert isinstance(result, IngestedPaper)
    assert result.paper_id == "p1"
    assert result.chunk_count >= 1

    hits = bm25.search("attention", top_k=5)
    assert hits and hits[0].chunk_id.startswith("p1::")

    [vector] = await embedder.embed_texts(["attention"])
    matches = await vectorstore.search(vector, top_k=5)
    assert matches


async def test_ingest_with_contextualizer_populates_context(tmp_path: Path) -> None:
    pdf_path = _make_pdf(tmp_path)
    paper = Paper(paper_id="p3", title="Ctx", pdf_path=pdf_path)
    embedder = FakeEmbedder(dim=8)
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="t3", dim=8)
    await vectorstore.ensure_collection()
    bm25 = Bm25Index()
    llm = _StubLLM(text="situating blurb")

    result = await ingest_paper(
        paper=paper,
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        contextualizer_llm=llm,
        contextualizer_model="cheap-model",
    )

    assert result.chunk_count >= 1
    assert llm.calls == result.chunk_count
    assert all(c.context == "situating blurb" for c in result.chunks)
    # BM25 should index the contextualized text (so blurb terms hit too).
    hits = bm25.search("situating", top_k=5)
    assert hits, "blurb terms should hit BM25 because indexed_text includes context"


async def test_ingest_empty_pdf_yields_zero_chunks(tmp_path: Path) -> None:
    doc = fitz.open()
    doc.new_page()
    pdf_path = tmp_path / "blank.pdf"
    doc.save(pdf_path)
    doc.close()

    paper = Paper(paper_id="p2", title="Blank", pdf_path=pdf_path)
    embedder = FakeEmbedder(dim=8)
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="t2", dim=8)
    await vectorstore.ensure_collection()
    bm25 = Bm25Index()

    result = await ingest_paper(paper=paper, embedder=embedder, vectorstore=vectorstore, bm25=bm25)
    assert result.chunk_count == 0
