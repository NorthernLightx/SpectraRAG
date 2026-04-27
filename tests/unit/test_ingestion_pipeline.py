"""Ingestion pipeline: paper → pages → chunks → indexed in BM25 + vectorstore."""

from __future__ import annotations

from pathlib import Path

import fitz

from src.ingestion.pipeline import IngestedPaper, ingest_paper
from src.rag.bm25 import Bm25Index
from src.rag.vectorstore import QdrantVectorStore
from src.types import Paper
from tests.fakes import FakeEmbedder


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

    result = await ingest_paper(
        paper=paper, embedder=embedder, vectorstore=vectorstore, bm25=bm25
    )

    assert isinstance(result, IngestedPaper)
    assert result.paper_id == "p1"
    assert result.chunk_count >= 1

    hits = bm25.search("attention", top_k=5)
    assert hits and hits[0].chunk_id.startswith("p1::")

    [vector] = await embedder.embed_texts(["attention"])
    matches = await vectorstore.search(vector, top_k=5)
    assert matches


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

    result = await ingest_paper(
        paper=paper, embedder=embedder, vectorstore=vectorstore, bm25=bm25
    )
    assert result.chunk_count == 0
