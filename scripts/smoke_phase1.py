"""End-to-end smoke for Phase 1.2/1.3: ingest a real PDF, query it, optionally answer.

Run: `uv run python -m scripts.smoke_phase1 --pdf data/papers/<file>.pdf "your query"`

Requires Docker services up: `docker compose up -d qdrant ollama`
and `docker exec rag-ollama ollama pull bge-m3` once.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.ingestion.pipeline import ingest_paper
from src.rag.bm25 import Bm25Index
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Paper, Query


async def main(
    *,
    pdf_path: Path,
    query_text: str,
    qdrant_url: str,
    ollama_url: str,
    top_k: int,
) -> None:
    print(f"[1/4] Embedding via Ollama at {ollama_url}")
    embedder = OllamaBgeEmbedder(base_url=ollama_url)

    print(f"[2/4] Connecting Qdrant at {qdrant_url}")
    vectorstore = QdrantVectorStore(
        url=qdrant_url, collection_name="smoke_phase1", dim=embedder.dim
    )
    await vectorstore.ensure_collection()
    bm25 = Bm25Index()

    print(f"[3/4] Ingesting {pdf_path}")
    paper = Paper(paper_id=pdf_path.stem, title=pdf_path.stem, pdf_path=pdf_path)
    ingested = await ingest_paper(
        paper=paper, embedder=embedder, vectorstore=vectorstore, bm25=bm25
    )
    print(f"      -> {ingested.chunk_count} chunks indexed")
    if ingested.chunk_count == 0:
        print("No chunks produced. Aborting.")
        return

    print(f"[4/4] Querying: {query_text!r} (top_k={top_k})")
    retriever = PipelineRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id={c.chunk_id: c for c in ingested.chunks},
    )
    results = await retriever.retrieve(Query(text=query_text, top_k=top_k))
    print(f"      -> {len(results)} results")
    for index, result in enumerate(results, start=1):
        section = result.metadata.get("section") or "?"
        snippet = result.text[:160].replace("\n", " ")
        print(
            f"  {index}. [{result.chunk_id}] (score={result.score:.4f}) "
            f"section={section!r}\n     {snippet}..."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end smoke for Phase 1 retrieval.")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to a PDF to ingest.")
    parser.add_argument("query", nargs="?", default="What does this paper introduce?")
    parser.add_argument("--qdrant", default=os.environ.get("RAG_QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--ollama", default=os.environ.get("RAG_OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(
        main(
            pdf_path=args.pdf,
            query_text=args.query,
            qdrant_url=args.qdrant,
            ollama_url=args.ollama,
            top_k=args.top_k,
        )
    )
