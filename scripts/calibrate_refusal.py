"""Calibrate `refusal_score_threshold` from rerank-score distributions.

Runs every golden query through the *text* retriever (matching the committed
baseline config: paper-id-filter + length-norm + region-number-boost) and
records the top rerank score. Splits results by category — out_of_corpus
queries should ideally have a lower top-score than answerable in-corpus
queries, so the right threshold separates them.

Usage:
    uv run python -m scripts.calibrate_refusal \\
        --collection eval_phase33d_vlm_lengthnorm \\
        --golden data/golden/v3.yaml

Why text-only: ADR 0008's regex classifier routes all 8 OOC queries in v3
to text-only (none of them mention 'Figure N' / 'Table N' / 'compare').
The Generator's refusal gate checks `RetrievalResult.score`, which for the
text leg is the bge-reranker-v2-m3 logit (post length-norm if enabled);
for the visual leg it would be the ColQwen2 MaxSim. Mixing the two scales
is the reason production should only check rerank scores from the text
leg or use a category-specific threshold; here we calibrate the text leg
threshold which covers all v3 OOC cases.

Out: a per-query JSON + a stdout summary suggesting a threshold.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from src.embeddings.ollama_bge import OllamaBgeEmbedder
from src.eval.golden_set import load_golden_set
from src.rag.bm25 import Bm25Index
from src.rag.rerank import BgeReranker
from src.rag.retrievers.pipeline import PipelineRetriever
from src.rag.retrievers.region_boost import RegionNumberBoostRetriever
from src.rag.vectorstore import QdrantVectorStore
from src.types import Query


async def _build_retriever(
    *,
    qdrant_url: str,
    collection: str,
    ollama_url: str,
    rerank_length_norm: bool,
    region_boost: bool,
) -> tuple[PipelineRetriever | RegionNumberBoostRetriever, int]:
    """Re-build the text retriever from an existing populated Qdrant collection.

    Returns the retriever and the number of chunks it sees (sanity check —
    if 0, the collection name is wrong or the schema is from an older run).
    """
    embedder = OllamaBgeEmbedder(base_url=ollama_url)
    vectorstore = QdrantVectorStore(url=qdrant_url, collection_name=collection, dim=embedder.dim)
    chunks = await vectorstore.scroll_chunks()
    if not chunks:
        raise SystemExit(f"Collection {collection!r} is empty or missing the required schema.")

    bm25 = Bm25Index()
    bm25.add(chunks)
    chunks_by_id = {c.chunk_id: c for c in chunks}

    reranker = BgeReranker(length_norm=rerank_length_norm)
    pipeline = PipelineRetriever(
        embedder=embedder,
        vectorstore=vectorstore,
        bm25=bm25,
        chunks_by_id=chunks_by_id,
        reranker=reranker,
    )
    if region_boost:
        return RegionNumberBoostRetriever(base=pipeline), len(chunks)
    return pipeline, len(chunks)


def _suggest_threshold(records: list[dict[str, Any]]) -> tuple[float, str]:
    """Pick a threshold separating OOC top-scores from in-corpus top-scores.

    Strategy: midpoint of (max OOC top-score) and (min in-corpus top-score).
    If they overlap, we report the overlap and pick a threshold that
    minimises misclassifications (greedy over candidates).
    """
    ooc: list[float] = sorted(
        float(r["max_score"])
        for r in records
        if r["category"] == "out_of_corpus" and r["max_score"] is not None
    )
    in_corpus: list[float] = sorted(
        float(r["max_score"])
        for r in records
        if r["category"] != "out_of_corpus" and r["max_score"] is not None
    )
    if not ooc or not in_corpus:
        return 0.0, "no usable scores in one of the two buckets; cannot suggest"

    max_ooc = max(ooc)
    min_in = min(in_corpus)
    if max_ooc < min_in:
        threshold = (max_ooc + min_in) / 2
        return threshold, f"clean separation: max(OOC)={max_ooc:.3f} < min(in-corpus)={min_in:.3f}"

    # Overlap: candidate thresholds are sorted unique scores; pick the one
    # that minimises (OOC above thresh) + (in-corpus below thresh).
    candidates = sorted(set(ooc) | set(in_corpus))
    best_thresh = candidates[0]
    best_err = len(records) + 1
    for t in candidates:
        ooc_above = sum(1 for s in ooc if s >= t)
        in_below = sum(1 for s in in_corpus if s < t)
        err = ooc_above + in_below
        if err < best_err:
            best_err = err
            best_thresh = t
    note = (
        f"overlap: max(OOC)={max_ooc:.3f}, min(in-corpus)={min_in:.3f}; "
        f"best threshold {best_thresh:.3f} misclassifies {best_err} of {len(records)}"
    )
    return best_thresh, note


async def _calibrate(
    *,
    golden_path: Path,
    qdrant_url: str,
    collection: str,
    ollama_url: str,
    top_k: int,
    rerank_length_norm: bool,
    region_boost: bool,
    paper_id_filter: bool,
    out_path: Path | None,
) -> list[dict[str, Any]]:
    retriever, n_chunks = await _build_retriever(
        qdrant_url=qdrant_url,
        collection=collection,
        ollama_url=ollama_url,
        rerank_length_norm=rerank_length_norm,
        region_boost=region_boost,
    )
    golden_set = load_golden_set(golden_path)
    print(
        f"Calibrating against {len(golden_set.queries)} queries from {golden_set.name}/{golden_set.version}"
    )
    print(f"Corpus: {n_chunks} chunks in Qdrant collection '{collection}'")
    print(
        f"Config: length_norm={rerank_length_norm}, region_boost={region_boost}, "
        f"paper_id_filter={paper_id_filter}"
    )
    print()

    records: list[dict[str, Any]] = []
    for q in golden_set.queries:
        filters: dict[str, str] = {}
        if paper_id_filter and q.paper_id:
            filters["paper_id"] = q.paper_id
        rag_query = Query(text=q.text, top_k=top_k, filters=filters)
        results = await retriever.retrieve(rag_query)
        if results:
            scores = [r.score for r in results]
            max_score = max(scores)
            top_chunk_id = results[0].chunk_id
        else:
            max_score = None
            top_chunk_id = None
        records.append(
            {
                "query_id": q.query_id,
                "category": q.category,
                "paper_id": q.paper_id,
                "text": q.text,
                "n_returned": len(results),
                "max_score": max_score,
                "top_chunk_id": top_chunk_id,
            }
        )

    # Print summary
    print(f"{'query_id':40s} {'category':14s} {'top_score':>10s}  top_chunk")
    print("-" * 100)
    for r in sorted(records, key=lambda x: (x["category"], -(x["max_score"] or -999))):
        score_str = f"{r['max_score']:.4f}" if r["max_score"] is not None else "(no results)"
        print(
            f"{r['query_id']:40s} {r['category']:14s} {score_str:>10s}  {r['top_chunk_id'] or ''}"
        )
    print()

    # Bucket aggregates
    by_cat: dict[str, list[float]] = {}
    for r in records:
        if r["max_score"] is None:
            continue
        by_cat.setdefault(str(r["category"]), []).append(float(r["max_score"]))
    print("Per-category top-score distribution:")
    for cat, scores in sorted(by_cat.items()):
        scores_sorted = sorted(scores)
        if scores_sorted:
            print(
                f"  {cat:15s} n={len(scores_sorted):2d} "
                f"min={scores_sorted[0]:.3f} "
                f"med={scores_sorted[len(scores_sorted) // 2]:.3f} "
                f"max={scores_sorted[-1]:.3f}"
            )
    print()

    threshold, note = _suggest_threshold(records)
    print(f"Suggested refusal_score_threshold: {threshold:.4f}")
    print(f"  ({note})")

    if out_path:
        out_path.write_text(
            json.dumps(
                {
                    "config": {
                        "collection": collection,
                        "rerank_length_norm": rerank_length_norm,
                        "region_boost": region_boost,
                        "paper_id_filter": paper_id_filter,
                        "top_k": top_k,
                    },
                    "suggested_threshold": threshold,
                    "suggested_threshold_note": note,
                    "per_query": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote per-query data to {out_path}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=Path("data/golden/v3.yaml"))
    parser.add_argument("--qdrant", default="http://localhost:6333")
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument(
        "--collection",
        default="eval_phase33d_vlm_lengthnorm",
        help="Qdrant collection holding the corpus to calibrate against. "
        "Default = the latest committed-baseline collection.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rerank-length-norm", action="store_true", default=True)
    parser.add_argument("--no-rerank-length-norm", dest="rerank_length_norm", action="store_false")
    parser.add_argument("--region-boost", action="store_true", default=True)
    parser.add_argument("--no-region-boost", dest="region_boost", action="store_false")
    parser.add_argument("--paper-id-filter", action="store_true", default=True)
    parser.add_argument("--no-paper-id-filter", dest="paper_id_filter", action="store_false")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/eval/runs/calibration-refusal.json"),
        help="Path to write the per-query JSON. Pass empty string to skip.",
    )
    args = parser.parse_args()

    asyncio.run(
        _calibrate(
            golden_path=args.golden,
            qdrant_url=args.qdrant,
            collection=args.collection,
            ollama_url=args.ollama,
            top_k=args.top_k,
            rerank_length_norm=args.rerank_length_norm,
            region_boost=args.region_boost,
            paper_id_filter=args.paper_id_filter,
            out_path=args.out if str(args.out) else None,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
