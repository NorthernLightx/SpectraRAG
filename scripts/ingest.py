"""Ingest a directory of PDFs into BM25 + Qdrant. Supports :memory: or remote.

Run: `uv run python -m scripts.ingest --pdf-dir data/papers --qdrant ":memory:"`
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.ingestion.pipeline import ingest_paper
from src.observability.logging import configure_logging
from src.rag.bm25 import Bm25Index
from src.rag.vectorstore import QdrantVectorStore
from src.types import Paper


async def main(*, pdf_dir: Path, qdrant_url: str, ollama_url: str, collection: str) -> None:
    embedder = OllamaBgeEmbedder(base_url=ollama_url)
    vectorstore = QdrantVectorStore(url=qdrant_url, collection_name=collection, dim=embedder.dim)
    await vectorstore.ensure_collection()
    bm25 = Bm25Index()

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return

    for pdf_path in pdfs:
        paper = Paper(paper_id=pdf_path.stem, title=pdf_path.stem, pdf_path=pdf_path)
        result = await ingest_paper(
            paper=paper, embedder=embedder, vectorstore=vectorstore, bm25=bm25
        )
        print(f"Ingested {paper.paper_id}: {result.chunk_count} chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into BM25 + Qdrant.")
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/papers"))
    parser.add_argument("--qdrant", default=":memory:", help="Qdrant URL or ':memory:'")
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--collection", default="papers_v1")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = Path("logs") / f"ingest-{timestamp}.log"
    configure_logging(level="INFO", env="local", log_file=log_file)
    print(f"Logging JSON to {log_file}")

    asyncio.run(
        main(
            pdf_dir=args.pdf_dir,
            qdrant_url=args.qdrant,
            ollama_url=args.ollama,
            collection=args.collection,
        )
    )
