"""Run a golden-set evaluation against a live Ollama+Qdrant corpus.

Run:
  uv run python -m scripts.eval_run --pdf data/papers/<file>.pdf \
      --golden data/golden/v1.yaml

Requires `docker compose up -d qdrant ollama` and `ollama pull bge-m3`.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.eval.golden_set import load_golden_set
from src.eval.report import write_run_json, write_run_markdown
from src.eval.runner import evaluate
from src.ingestion.pipeline import ingest_paper
from src.observability.logging import configure_logging, get_logger
from src.rag.bm25 import Bm25Index
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Paper


async def _main(
    *,
    pdf_path: Path,
    golden_path: Path,
    qdrant_url: str,
    ollama_url: str,
    output_dir: Path,
    top_k: int,
    collection: str,
) -> None:
    log = get_logger("scripts.eval_run")
    log.info("eval_cli.start", pdf=str(pdf_path), golden=str(golden_path))

    embedder = OllamaBgeEmbedder(base_url=ollama_url)
    vectorstore = QdrantVectorStore(
        url=qdrant_url, collection_name=collection, dim=embedder.dim
    )
    await vectorstore.ensure_collection()
    bm25 = Bm25Index()

    paper = Paper(paper_id=pdf_path.stem, title=pdf_path.stem, pdf_path=pdf_path)
    ingested = await ingest_paper(
        paper=paper, embedder=embedder, vectorstore=vectorstore, bm25=bm25
    )
    print(f"Ingested {ingested.chunk_count} chunks from {pdf_path.name}")

    retriever = PipelineRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id={c.chunk_id: c for c in ingested.chunks},
    )

    golden_set = load_golden_set(golden_path)
    print(f"Loaded golden set {golden_set.name} {golden_set.version} ({len(golden_set.queries)} queries)")

    run = await evaluate(
        retriever=retriever,
        golden_set=golden_set,
        top_k=top_k,
        config={
            "retriever": "pipeline",
            "rerank": False,
            "top_k": top_k,
            "paper_id": paper.paper_id,
            "embedding_model": "bge-m3",
            "embedding_dim": embedder.dim,
        },
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"run-{timestamp}.json"
    md_path = output_dir / f"run-{timestamp}.md"
    write_run_json(run, json_path)
    write_run_markdown(run, md_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    log.info("eval_cli.done", run_id=run.run_id, json=str(json_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a golden-set evaluation end-to-end.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=Path("data/golden/v1.yaml"))
    parser.add_argument("--qdrant", default="http://localhost:6333")
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval/runs"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--collection", default="eval_phase1")
    args = parser.parse_args()

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    log_file = Path("logs") / f"eval-{timestamp}.log"
    configure_logging(level="INFO", env="local", log_file=log_file)
    print(f"Logging JSON to {log_file}")

    asyncio.run(
        _main(
            pdf_path=args.pdf,
            golden_path=args.golden,
            qdrant_url=args.qdrant,
            ollama_url=args.ollama,
            output_dir=args.output_dir,
            top_k=args.top_k,
            collection=args.collection,
        )
    )
