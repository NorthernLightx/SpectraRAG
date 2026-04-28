"""End-to-end: synthetic PDF → ingest → PipelineRetriever wired into /query."""

from __future__ import annotations

from pathlib import Path

import fitz
from fastapi.testclient import TestClient

from src.api.deps import get_retriever
from src.api.main import create_app
from src.ingestion.pipeline import ingest_paper
from src.rag.bm25 import Bm25Index
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Paper
from tests.fakes import FakeEmbedder


def _make_pdf(tmp_path: Path) -> Path:
    """Create a multi-page PDF with content the query should match."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "1 Introduction\n\nThe transformer architecture revolutionised natural language processing.",
        fontsize=11,
    )
    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "2 Method\n\nWe extend transformers with attention over retrieved passages.",
        fontsize=11,
    )
    pdf_path = tmp_path / "transformer-paper.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


async def test_full_pipeline_end_to_end(tmp_path: Path) -> None:
    pdf_path = _make_pdf(tmp_path)
    paper = Paper(paper_id="ts1", title="Transformer survey", pdf_path=pdf_path)

    embedder = FakeEmbedder(dim=8)
    vectorstore = QdrantVectorStore(url=":memory:", collection_name="e2e", dim=8)
    await vectorstore.ensure_collection()
    bm25 = Bm25Index()

    ingested = await ingest_paper(
        paper=paper, embedder=embedder, vectorstore=vectorstore, bm25=bm25
    )
    assert ingested.chunk_count >= 2

    retriever = PipelineRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id={c.chunk_id: c for c in ingested.chunks},
    )

    app = create_app(log_file=None)
    app.dependency_overrides[get_retriever] = lambda: retriever

    with TestClient(app) as client:
        response = client.post("/query", json={"text": "transformer attention", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert len(body) >= 1
    assert all(item["paper_id"] == "ts1" for item in body)
    assert any("transformer" in item["text"].lower() for item in body)
