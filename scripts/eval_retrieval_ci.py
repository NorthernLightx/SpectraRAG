"""CPU-only page-level text-retrieval eval against the committed qdrant_local snapshot.

The per-commit CI regression gate. No Ollama, no GPU, no LLM: opens the baked
`rag_corpus` snapshot in embedded path-mode, embeds golden queries with bge-m3
on CPU (the same sentence-transformers backend the Cloud Run image uses), runs
the hybrid dense+BM25+RRF PipelineRetriever, and writes retrieval metrics
(nDCG@5, recall@10, MRR) as an EvalRun JSON that scripts.check_regression can
gate against a committed retrieval baseline.

Metrics are scored at PAGE granularity: both the golden `relevant_chunk_ids`
and the retrieved chunk ids are projected to their `paper::pN` page before
scoring. `rag_corpus` is the shipped demo corpus and is periodically re-baked
by the docling chunker (ADR 0017 / 0021), which renumbers the `::cN` chunk
suffix, so the v3 golden's chunk-level labels drift out of sync with it while
the page they point at does not. Page projection coarsens the *existing* human
labels (it authors no new ground truth) and is re-chunk-robust, the same reason
ADR 0019's answer_correctness judges answer text rather than chunk ids.

Deliberately excludes the visual router leg (ColQwen2, GPU) and the LLM
generation/judge legs (API cost, sampling variance). Those run in the manual /
scheduled full eval (scripts/eval_run.py) and rebaseline data/eval/baseline.json.
This gate covers the deterministic slice a stock CPU runner can reproduce.

Run:
  uv run python -m scripts.eval_retrieval_ci \
      --golden data/golden/v3.yaml \
      --output data/eval/baseline_retrieval.json
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from src.embeddings.sentence_transformers_bge import SentenceTransformersBgeEmbedder
from src.eval.golden_set import load_golden_set
from src.eval.metrics_retrieval import ndcg_at_k, recall_at_k, reciprocal_rank
from src.eval.report import write_run_json
from src.observability.logging import configure_logging, get_logger
from src.rag.bm25 import Bm25Index
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import EvalRun, PerQueryResult, Query, RetrievalMetrics


def _page(chunk_id: str) -> str:
    """Project `paper::pN::cX` (or `paper::pN::page`) down to `paper::pN`."""
    parts = chunk_id.split("::")
    return "::".join(parts[:2]) if len(parts) >= 2 else chunk_id


async def _main(
    *, snapshot: Path, collection: str, golden_path: Path, output: Path, top_k: int
) -> None:
    log = get_logger("scripts.eval_retrieval_ci")
    embedder = SentenceTransformersBgeEmbedder(device="cpu")
    vectorstore = QdrantVectorStore(
        url=f"path:{snapshot}", collection_name=collection, dim=embedder.dim
    )
    chunks = await vectorstore.scroll_chunks()
    if not chunks:
        raise SystemExit(
            f"snapshot {snapshot}/{collection!r} has no chunks: wrong path, or a "
            "pre-payload-schema collection. Re-bake with scripts/bootstrap_corpus.py."
        )
    chunks_by_id = {c.chunk_id: c for c in chunks}
    paper_ids = sorted({c.paper_id for c in chunks})
    bm25 = Bm25Index()
    bm25.add(chunks)
    print(f"Loaded {len(chunks)} chunks across {len(paper_ids)} papers from {collection!r}")

    retriever = PipelineRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id=chunks_by_id,
        reranker=None,
    )

    golden_set = load_golden_set(golden_path)
    print(
        f"Loaded golden set {golden_set.name} {golden_set.version} "
        f"({len(golden_set.queries)} queries)"
    )

    started_at = datetime.now(UTC)
    per_query: list[PerQueryResult] = []
    for q in golden_set.queries:
        # paper_id_filter: scope retrieval to the query's source paper, mirroring
        # how the repo evaluates retrieval (ADR 0009 follow-up). Production
        # callers pass no paper hint; this is an eval-only fairness knob.
        filters = {"paper_id": q.paper_id} if q.paper_id else {}
        started = time.monotonic()
        results = await retriever.retrieve(Query(text=q.text, top_k=top_k, filters=filters))
        latency_ms = int((time.monotonic() - started) * 1000)
        retrieved_ids = [r.chunk_id for r in results]
        # Dedup projected pages preserving rank so nDCG/MRR see the position of
        # the first relevant page, not the first relevant chunk.
        relevant_pages = list(dict.fromkeys(_page(c) for c in q.relevant_chunk_ids))
        retrieved_pages = list(dict.fromkeys(_page(c) for c in retrieved_ids))
        per_query.append(
            PerQueryResult(
                query_id=q.query_id,
                category=q.category,
                text=q.text,
                retrieved_chunk_ids=retrieved_ids,
                retrieval=RetrievalMetrics(
                    ndcg_at_5=ndcg_at_k(relevant_pages, retrieved_pages, k=5),
                    recall_at_10=recall_at_k(relevant_pages, retrieved_pages, k=10),
                    mrr=reciprocal_rank(relevant_pages, retrieved_pages),
                ),
                latency_ms=latency_ms,
            )
        )
    run = EvalRun(
        run_id=uuid4().hex[:12],
        started_at=started_at,
        finished_at=datetime.now(UTC),
        golden_set_name=golden_set.name,
        golden_set_version=golden_set.version,
        config={
            "retriever": "pipeline-text-cpu",
            "granularity": "page",
            "rerank": False,
            "router": False,
            "generate": False,
            "judge": False,
            "embedding_model": "bge-m3",
            "embedding_dim": embedder.dim,
            "embedding_backend": "sentence-transformers-cpu",
            "top_k": top_k,
            "paper_id_filter": True,
            "exclude_decoration": True,
            "snapshot": str(snapshot),
            "collection": collection,
            "paper_ids": paper_ids,
        },
        per_query=per_query,
    )
    write_run_json(run, output)
    log.info("eval_retrieval_ci.done", run_id=run.run_id, output=str(output))
    print(f"Wrote {output} (run_id={run.run_id})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU text-retrieval eval for the CI gate.")
    parser.add_argument("--snapshot", type=Path, default=Path("qdrant_local"))
    parser.add_argument("--collection", default="rag_corpus")
    parser.add_argument("--golden", type=Path, default=Path("data/golden/v3.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/runs/retrieval-ci.json"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    configure_logging(level="WARNING", env="local")
    asyncio.run(
        _main(
            snapshot=args.snapshot,
            collection=args.collection,
            golden_path=args.golden,
            output=args.output,
            top_k=args.top_k,
        )
    )
