"""End-to-end ingestion: paper → pages → chunks → indexed in BM25 + vectorstore."""

from __future__ import annotations

from dataclasses import dataclass

from src.embeddings.protocol import Embedder
from src.ingestion.chunking import chunk_pages
from src.ingestion.pdf import extract_pages
from src.observability.logging import get_logger, timed_event
from src.rag.bm25 import Bm25Index
from src.rag.vectorstore import QdrantVectorStore
from src.types import Chunk, Paper

_log = get_logger(__name__)


@dataclass(frozen=True)
class IngestedPaper:
    """Outcome of ingesting one paper."""

    paper_id: str
    chunk_count: int
    chunks: list[Chunk]


async def ingest_paper(
    *,
    paper: Paper,
    embedder: Embedder,
    vectorstore: QdrantVectorStore,
    bm25: Bm25Index,
    target_chars: int = 1200,
    overlap_chars: int = 200,
) -> IngestedPaper:
    """Full pipeline: extract pages, chunk, embed, index in vector store + BM25."""
    with timed_event(
        _log, "ingest.done", paper_id=paper.paper_id, pdf_path=str(paper.pdf_path)
    ) as ctx:
        pages = extract_pages(paper_id=paper.paper_id, pdf_path=paper.pdf_path)
        chunks = chunk_pages(pages, target_chars=target_chars, overlap_chars=overlap_chars)
        ctx["pages"] = len(pages)
        ctx["chunks"] = len(chunks)
        if not chunks:
            ctx["embedding_dim"] = 0
            return IngestedPaper(paper_id=paper.paper_id, chunk_count=0, chunks=[])

        embeddings = await embedder.embed_texts([c.text for c in chunks])
        await vectorstore.upsert_chunks(chunks, embeddings)
        bm25.add(chunks)
        ctx["embedding_dim"] = len(embeddings[0]) if embeddings else 0
        return IngestedPaper(paper_id=paper.paper_id, chunk_count=len(chunks), chunks=chunks)
