"""Run a golden-set evaluation against the visual retriever (ColQwen2).

Sibling to `scripts/eval_run.py`. Kept separate because ColQwen2 needs the
GPU exclusively (~5 GB VRAM); cohabiting with bge-m3 + reranker would
trigger CPU fallback. The flow:

  1. Render each PDF page to PNG (idempotent).
  2. Embed every page once into ColQwen2 multi-vector tensors.
  3. For each golden query: embed the query, MaxSim-score against every
     page, return top-K.
  4. Compute retrieval metrics against the golden set's `relevant_pages`
     (NOT `relevant_chunk_ids`, which target text chunks).
  5. Optionally generate + judge using a separate LLM client.

Run:
  uv run python -m scripts.eval_visual \\
      --pdf data/papers/2604.22753v1.pdf data/papers/2604.28180v1.pdf ... \\
      --golden data/golden/v3.yaml \\
      --output-dir data/eval/runs

Prerequisites: a CUDA-capable PyTorch (the rerank GPU unlock from earlier),
ColQwen2 weights downloadable from HuggingFace (~7 GB on first run).
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from src.eval.golden_set import load_golden_set
from src.eval.metrics_retrieval import ndcg_at_k, recall_at_k, reciprocal_rank
from src.eval.report import write_run_json, write_run_markdown
from src.ingestion.visual import render_pages
from src.observability.logging import configure_logging, get_logger
from src.rag.retrievers.visual import build_visual_retriever
from src.types import (
    EvalRun,
    GoldenQuery,
    PerQueryResult,
    Query,
    RetrievalMetrics,
)


def _page_chunk_id(paper_id: str, page_no: int) -> str:
    return f"{paper_id}::p{page_no}::page"


def _expected_page_chunks(query: GoldenQuery) -> list[str]:
    """Translate `relevant_pages` to visual page chunk-ids. Falls back to
    parsing chunk-ids of the form `paper::pN::cM` if `relevant_pages` is empty."""
    if query.relevant_pages:
        return [_page_chunk_id(query.paper_id, p) for p in query.relevant_pages]
    pages: set[int] = set()
    for cid in query.relevant_chunk_ids:
        # `<paper>::p<page>::c<chunk>`
        parts = cid.split("::")
        for part in parts:
            if part.startswith("p") and part[1:].isdigit():
                pages.add(int(part[1:]))
                break
    return [_page_chunk_id(query.paper_id, p) for p in sorted(pages)]


async def _main(
    *,
    pdf_paths: list[Path],
    golden_path: Path,
    output_dir: Path,
    pages_dir: Path,
    top_k: int,
    dpi: int,
    model_name: str,
    device: str,
) -> None:
    log = get_logger("scripts.eval_visual")
    log.info(
        "visual_eval.start",
        pdfs=[str(p) for p in pdf_paths],
        golden=str(golden_path),
        model=model_name,
    )

    pages_by_paper: dict[str, list[tuple[int, Path]]] = {}
    for pdf_path in pdf_paths:
        paper_id = pdf_path.stem
        rendered = render_pages(paper_id, pdf_path, out_dir=pages_dir, dpi=dpi)
        pages_by_paper[paper_id] = [(p.page_number, p.image_path) for p in rendered]
        print(f"Rendered {len(rendered)} pages for {paper_id}")

    print(f"Loading {model_name} and embedding pages on {device}...")
    retriever = await build_visual_retriever(pages_by_paper, model_name=model_name, device=device)

    golden_set = load_golden_set(golden_path)
    print(
        f"Loaded golden set {golden_set.name} {golden_set.version} ({len(golden_set.queries)} queries)"
    )

    started_at = datetime.now(UTC)
    per_query: list[PerQueryResult] = []
    for query in golden_set.queries:
        started = time.monotonic()
        relevant_chunks = _expected_page_chunks(query)
        retrieved = await retriever.retrieve(Query(text=query.text, top_k=top_k))
        retrieved_ids = [r.chunk_id for r in retrieved]
        per_query.append(
            PerQueryResult(
                query_id=query.query_id,
                category=query.category,
                text=query.text,
                retrieved_chunk_ids=retrieved_ids,
                retrieval=RetrievalMetrics(
                    ndcg_at_5=ndcg_at_k(relevant_chunks, retrieved_ids, k=5),
                    recall_at_10=recall_at_k(relevant_chunks, retrieved_ids, k=10),
                    mrr=reciprocal_rank(relevant_chunks, retrieved_ids),
                ),
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        )
        print(
            f"  {query.query_id}: ndcg5={per_query[-1].retrieval.ndcg_at_5:.3f} "
            f"recall10={per_query[-1].retrieval.recall_at_10:.3f} "
            f"top={retrieved_ids[0] if retrieved_ids else '-'}"
        )

    finished_at = datetime.now(UTC)
    run = EvalRun(
        run_id=uuid4().hex[:12],
        started_at=started_at,
        finished_at=finished_at,
        golden_set_name=golden_set.name,
        golden_set_version=golden_set.version,
        config={
            "retriever": "visual",
            "model": model_name,
            "device": device,
            "top_k": top_k,
            "dpi": dpi,
            "paper_ids": [p.stem for p in pdf_paths],
            "embedding_model": model_name,
        },
        per_query=per_query,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"run-visual-{timestamp}.json"
    md_path = output_dir / f"run-visual-{timestamp}.md"
    write_run_json(run, json_path)
    write_run_markdown(run, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    log.info("visual_eval.done", run_id=run.run_id, json=str(json_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True, nargs="+")
    parser.add_argument("--golden", type=Path, default=Path("data/golden/v3.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval/runs"))
    parser.add_argument("--pages-dir", type=Path, default=Path("data/pages"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--model", default="vidore/colqwen2-v1.0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = Path("logs") / f"eval-visual-{timestamp}.log"
    configure_logging(level="INFO", env="local", log_file=log_file)
    print(f"Logging JSON to {log_file}")

    asyncio.run(
        _main(
            pdf_paths=args.pdf,
            golden_path=args.golden,
            output_dir=args.output_dir,
            pages_dir=args.pages_dir,
            top_k=args.top_k,
            dpi=args.dpi,
            model_name=args.model,
            device=args.device,
        )
    )
