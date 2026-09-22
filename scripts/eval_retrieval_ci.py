"""CPU page-level retrieval eval of the served stack: the per-commit CI gate.

Builds the retriever the Cloud Run image serves, from the same `cpu` profile
(`src/config/profiles/cpu.yaml`: bge-m3 through sentence-transformers, BM25,
RRF, the MiniLM reranker), over the committed `rag_corpus` snapshot. No Ollama,
no GPU, no LLM.

The visual leg needs ColQwen2 and a page index that is not in git, so it is
replayed from a recorded fixture (`scripts/record_visual_legs.py`). The router
runs live on top of it, so fusion and routing are gated too. One pass writes
both arms: the hybrid arm, and the text arm read off the same run's text leg.

Metrics are scored at PAGE granularity: both the golden `relevant_chunk_ids`
and the retrieved chunk ids are projected to their `paper::pN` page before
scoring. `rag_corpus` is the shipped demo corpus and is periodically re-baked
by the docling chunker (ADR 0017 / 0021), which renumbers the `::cN` chunk
suffix, so the v3 golden's chunk-level labels drift out of sync with it while
the page they point at does not. Page projection coarsens the *existing* human
labels (it authors no new ground truth) and is re-chunk-robust, the same reason
ADR 0019's answer_correctness judges answer text rather than chunk ids.

Run:
  uv run python -m scripts.eval_retrieval_ci \\
      --output data/eval/runs/retrieval-ci.json \\
      --hybrid-output data/eval/runs/retrieval-ci-hybrid.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.config.settings import load_settings
from src.eval.golden_set import load_golden_set
from src.eval.metrics_retrieval import ndcg_at_k, recall_at_k, reciprocal_rank
from src.eval.replay import ReplayRetriever
from src.eval.report import write_run_json
from src.observability.logging import configure_logging, get_logger
from src.rag.bm25 import Bm25Index
from src.rag.retrieval_config import (
    RetrievalConfig,
    build_embedder,
    build_routing_retriever,
    build_text_retriever,
)
from src.rag.retrievers.protocol import Retriever
from src.rag.retrievers.routing import get_last_leg_ids, get_last_routing_info, reset_last_leg_ids
from src.rag.vectorstore import QdrantVectorStore
from src.types import EvalRun, GoldenQuery, PerQueryResult, Query, RetrievalMetrics


def _page(chunk_id: str) -> str:
    """Project `paper::pN::cX` (or `paper::pN::page`) down to `paper::pN`."""
    parts = chunk_id.split("::")
    return "::".join(parts[:2]) if len(parts) >= 2 else chunk_id


def _scored(q: GoldenQuery, retrieved_ids: list[str], latency_ms: int) -> PerQueryResult:
    # Dedup projected pages preserving rank so nDCG/MRR see the position of
    # the first relevant page, not the first relevant chunk.
    relevant_pages = list(dict.fromkeys(_page(c) for c in q.relevant_chunk_ids))
    retrieved_pages = list(dict.fromkeys(_page(c) for c in retrieved_ids))
    return PerQueryResult(
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


async def _main(
    *,
    snapshot: Path,
    collection: str,
    golden_path: Path,
    output: Path,
    hybrid_output: Path | None,
    visual_legs: Path,
    profile: str,
    top_k: int,
) -> None:
    log = get_logger("scripts.eval_retrieval_ci")
    settings = load_settings(profile=profile)
    text_config = RetrievalConfig.from_settings(settings).text_only()
    embedder = build_embedder(text_config, ollama_url=settings.ollama_base_url)
    vectorstore = QdrantVectorStore(
        url=f"path:{snapshot}", collection_name=collection, dim=embedder.dim
    )
    chunks = await vectorstore.scroll_chunks()
    if not chunks:
        raise SystemExit(
            f"snapshot {snapshot}/{collection!r} has no chunks: wrong path, or a "
            "pre-payload-schema collection. Re-bake with scripts/bootstrap_corpus.py."
        )
    paper_ids = sorted({c.paper_id for c in chunks})
    bm25 = Bm25Index()
    bm25.add(chunks)
    print(f"Loaded {len(chunks)} chunks across {len(paper_ids)} papers from {collection!r}")

    text_retriever = build_text_retriever(
        text_config,
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id={c.chunk_id: c for c in chunks},
        reranker_device="cpu",
    )
    retriever: Retriever = text_retriever
    hybrid_config: RetrievalConfig | None = None
    fixture: dict[str, Any] = {}
    if hybrid_output is not None:
        fixture = json.loads(visual_legs.read_text(encoding="utf-8"))
        hybrid_config = RetrievalConfig.from_settings(
            settings.model_copy(
                update={"enable_multimodal": True, "visual_model": fixture["visual_model"]}
            )
        )
        retriever = build_routing_retriever(
            hybrid_config, text=text_retriever, visual=ReplayRetriever.from_fixture(visual_legs)
        )
    print(f"Text arm {text_config.fingerprint()}", end="")
    print(f", hybrid arm {hybrid_config.fingerprint()}" if hybrid_config else "")

    golden_set = load_golden_set(golden_path)
    print(
        f"Loaded golden set {golden_set.name} {golden_set.version} "
        f"({len(golden_set.queries)} queries)"
    )

    started_at = datetime.now(UTC)
    text_rows: list[PerQueryResult] = []
    hybrid_rows: list[PerQueryResult] = []
    for q in golden_set.queries:
        # paper_id_filter: scope retrieval to the query's source paper, mirroring
        # how the repo evaluates retrieval (ADR 0009 follow-up). Production
        # callers pass no paper hint; this is an eval-only fairness knob.
        filters = {"paper_id": q.paper_id} if q.paper_id else {}
        reset_last_leg_ids()
        started = time.monotonic()
        results = await retriever.retrieve(Query(text=q.text, top_k=top_k, filters=filters))
        latency_ms = int((time.monotonic() - started) * 1000)
        ids = [r.chunk_id for r in results]
        if hybrid_config is None:
            text_rows.append(_scored(q, ids, latency_ms))
            continue
        info = get_last_routing_info()
        if info is None or info.visual_failed:
            # The router falls back to text on a visual failure; in the gate that
            # would score the text arm as hybrid and hide a stale fixture.
            raise SystemExit(f"visual leg failed on {q.query_id}; re-record {visual_legs}")
        legs = get_last_leg_ids() or {}
        text_rows.append(_scored(q, legs.get("text", [])[:top_k], latency_ms))
        hybrid_rows.append(_scored(q, ids, latency_ms))
    finished_at = datetime.now(UTC)

    def run(rows: list[PerQueryResult], config: RetrievalConfig, arm: str) -> EvalRun:
        run_config: dict[str, Any] = {
            "retriever": f"ci-{arm}",
            "granularity": "page",
            "profile": profile,
            "retrieval_config": config.as_dict(),
            "retrieval_fingerprint": config.fingerprint(),
            "top_k": top_k,
            "paper_id_filter": True,
            "snapshot": str(snapshot),
            "collection": collection,
            "paper_ids": paper_ids,
        }
        if arm == "hybrid":
            run_config["visual_legs"] = {
                key: fixture[key] for key in ("golden", "visual_model", "dtype", "n_pages", "depth")
            }
        return EvalRun(
            run_id=uuid4().hex[:12],
            started_at=started_at,
            finished_at=finished_at,
            golden_set_name=golden_set.name,
            golden_set_version=golden_set.version,
            config=run_config,
            per_query=rows,
        )

    write_run_json(run(text_rows, text_config, "text"), output)
    print(f"Wrote {output}")
    if hybrid_output is not None and hybrid_config is not None:
        write_run_json(run(hybrid_rows, hybrid_config, "hybrid"), hybrid_output)
        print(f"Wrote {hybrid_output}")
    log.info("eval_retrieval_ci.done", output=str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU retrieval eval for the CI gate.")
    parser.add_argument("--snapshot", type=Path, default=Path("qdrant_local"))
    parser.add_argument("--collection", default="rag_corpus")
    parser.add_argument("--golden", type=Path, default=Path("data/golden/v3.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/runs/retrieval-ci.json"))
    parser.add_argument(
        "--hybrid-output",
        type=Path,
        default=None,
        help="Also run the hybrid router against the recorded visual leg and write it here.",
    )
    parser.add_argument(
        "--visual-legs", type=Path, default=Path("data/eval/fixtures/visual-legs-v3.json")
    )
    parser.add_argument("--profile", default="cpu", help="Settings profile of the served stack.")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    configure_logging(level="WARNING", env="local")
    asyncio.run(
        _main(
            snapshot=args.snapshot,
            collection=args.collection,
            golden_path=args.golden,
            output=args.output,
            hybrid_output=args.hybrid_output,
            visual_legs=args.visual_legs,
            profile=args.profile,
            top_k=args.top_k,
        )
    )
