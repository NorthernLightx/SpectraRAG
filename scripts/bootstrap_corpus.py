"""Bootstrap the corpus into a configured Qdrant.

Runs once per deploy / fresh local setup: walks --pdf-dir, ingests each PDF
through src.ingestion.pipeline.ingest_paper, and writes the resulting chunks
+ embeddings to the named Qdrant collection. Idempotent — refuses to re-ingest
if the collection already has points (override with --force).

Runs locally against `docker compose up qdrant ollama` for development parity.

Caveats:

  * BM25 lives in process memory (`src/rag/bm25.py`) — this script populates
    Qdrant but its BM25 dies with the process. The FastAPI app would need to
    rebuild BM25 from scratch at startup (or persist it separately). Eval
    scripts rebuild BM25 per run.
  * Same for `chunks_by_id` — the PipelineRetriever needs a chunk dict to
    resolve text after Qdrant returns chunk-ids. This script doesn't
    persist that either; eval / retrieval workflows materialize chunks
    fresh per run.

Usage:

    .venv/Scripts/python.exe -m scripts.bootstrap_corpus \\
        --pdf-dir data/papers \\
        --qdrant http://localhost:6333 \\
        --ollama http://localhost:11434 \\
        --collection rag_corpus

Pass `--force` to re-ingest into a non-empty collection (drops + recreates).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.ingestion.pipeline import ingest_paper
from src.observability.logging import configure_logging, get_logger
from src.rag.bm25 import Bm25Index
from src.rag.vectorstore import QdrantVectorStore
from src.types import Paper


async def _main(
    *,
    pdf_dir: Path,
    qdrant_url: str,
    ollama_url: str,
    collection: str,
    force: bool,
) -> None:
    log = get_logger("scripts.bootstrap_corpus")
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No .pdf files found in {pdf_dir}")
    print(f"Found {len(pdf_paths)} PDFs in {pdf_dir}")

    embedder = OllamaBgeEmbedder(base_url=ollama_url)
    vectorstore = QdrantVectorStore(url=qdrant_url, collection_name=collection, dim=embedder.dim)

    existing = await vectorstore.count()
    if existing > 0 and not force:
        print(
            f"Collection {collection!r} already has {existing} points — "
            f"skipping ingestion (pass --force to re-ingest)."
        )
        log.info("bootstrap.skip", collection=collection, existing_points=existing, force=force)
        return

    if existing > 0 and force:
        print(f"--force: dropping existing {existing} points in {collection!r}")

    await vectorstore.ensure_collection()
    bm25 = Bm25Index()  # in-process, throwaway — see module docstring caveats

    total_chunks = 0
    for pdf_path in pdf_paths:
        paper = Paper(paper_id=pdf_path.stem, title=pdf_path.stem, pdf_path=pdf_path)
        ingested = await ingest_paper(
            paper=paper,
            embedder=embedder,
            vectorstore=vectorstore,
            bm25=bm25,
            contextualizer_llm=None,
            contextualizer_model=None,
            contextualizer_concurrency=4,
            extract_figures_enabled=False,
            extract_tables_enabled=False,
            vlm_captioner=None,
        )
        total_chunks += ingested.chunk_count
        print(f"  {pdf_path.name}: {ingested.chunk_count} chunks")

    final = await vectorstore.count()
    print(
        f"\nIngested {total_chunks} chunks across {len(pdf_paths)} papers into "
        f"{collection!r} (collection now contains {final} points)"
    )
    log.info(
        "bootstrap.done",
        collection=collection,
        papers=len(pdf_paths),
        chunks_ingested=total_chunks,
        collection_size=final,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Idempotently ingest every PDF in --pdf-dir into a Qdrant collection."
    )
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/papers"))
    parser.add_argument("--qdrant", default="http://localhost:6333")
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--collection", default="rag_corpus")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if the collection already contains points.",
    )
    args = parser.parse_args()

    configure_logging(level="INFO", env="local", log_file=None)
    asyncio.run(
        _main(
            pdf_dir=args.pdf_dir,
            qdrant_url=args.qdrant,
            ollama_url=args.ollama,
            collection=args.collection,
            force=args.force,
        )
    )
