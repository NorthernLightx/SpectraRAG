"""End-to-end ingestion: paper → pages → chunks → indexed in BM25 + vectorstore."""

from __future__ import annotations

from dataclasses import dataclass

from src.embeddings.protocol import Embedder
from src.ingestion.chunking import chunk_pages
from src.ingestion.pdf import extract_pages
from src.rag.bm25 import Bm25Index
from src.rag.vectorstore import QdrantVectorStore
from src.types import Chunk, Paper


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
    pages = extract_pages(paper_id=paper.paper_id, pdf_path=paper.pdf_path)
    chunks = chunk_pages(pages, target_chars=target_chars, overlap_chars=overlap_chars)
    if not chunks:
        return IngestedPaper(paper_id=paper.paper_id, chunk_count=0, chunks=[])

    embeddings = await embedder.embed_texts([c.text for c in chunks])
    await vectorstore.upsert_chunks(chunks, embeddings)
    bm25.add(chunks)

    return IngestedPaper(paper_id=paper.paper_id, chunk_count=len(chunks), chunks=chunks)
