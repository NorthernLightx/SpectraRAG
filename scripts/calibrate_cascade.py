"""Calibrate `cascade_confidence_threshold` from text-only score distributions.

ADR 0010. The cascade router skips the visual leg when the text leg's top-1
rerank score >= threshold. To pick a good threshold we want:

  * High enough that "obvious" hybrid queries (figure/table/multi_hop) fall
    through and still get the visual leg's help.
  * Low enough that "obvious" text queries (factual/definitional with a sharp
    rerank winner) skip visual and save the ColQwen2 latency.

Strategy: run every golden query through the text leg only (the same code
path the cascade uses for its first pass), record per-category score
distributions, and suggest a threshold at the boundary where
{figure, table, multi_hop} starts and {factual, definitional} ends.

Usage:
    uv run python -m scripts.calibrate_cascade \\
        --collection eval_phase33d_vlm_lengthnorm \\
        --golden data/golden/v3.yaml

Out: a per-query JSON + a summary suggesting a threshold.
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
from src.rag.retrievers.routing import classify_query
from src.rag.vectorstore import QdrantVectorStore
from src.types import Query

_HYBRID_CATEGORIES = {"figure", "table", "multi_hop"}
_TEXT_CATEGORIES = {"factual", "definitional", "equation"}


async def _build_text_retriever(
    *,
    qdrant_url: str,
    collection: str,
    ollama_url: str,
    rerank_length_norm: bool,
    region_boost: bool,
) -> tuple[PipelineRetriever | RegionNumberBoostRetriever, int]:
    embedder = OllamaBgeEmbedder(base_url=ollama_url)
    vectorstore = QdrantVectorStore(url=qdrant_url, collection_name=collection, dim=embedder.dim)
    chunks = await vectorstore.scroll_chunks()
    if not chunks:
        raise SystemExit(f"Collection {collection!r} is empty or missing the schema.")
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
    """Pick threshold = midpoint of (min text-cat score) and (max hybrid-cat score).

    The router's category mode decides which queries get the visual leg. The
    cascade should mirror that decision, but using top-1 score as the gate.
    Specifically: cascade should fire visual on hybrid-category queries even
    when their text confidence is high. The suggestion below picks a
    conservative midpoint; the operator can tune up or down depending on
    whether they prefer "save more cost" (lower threshold = visual fires
    less) or "preserve hybrid quality" (higher threshold = visual fires more).
    """
    text_scores = [
        float(r["max_score"])
        for r in records
        if r["category"] in _TEXT_CATEGORIES and r["max_score"] is not None
    ]
    hybrid_scores = [
        float(r["max_score"])
        for r in records
        if r["category"] in _HYBRID_CATEGORIES and r["max_score"] is not None
    ]
    if not text_scores or not hybrid_scores:
        return 0.5, "insufficient data; falling back to 0.5"
    median_hybrid = sorted(hybrid_scores)[len(hybrid_scores) // 2]
    median_text = sorted(text_scores)[len(text_scores) // 2]
    threshold = max(0.0, min(1.0, median_hybrid + 0.05))
    note = (
        f"hybrid category median top-score = {median_hybrid:.3f}, "
        f"text category median = {median_text:.3f}; "
        f"set threshold just above hybrid median so most hybrid-category queries "
        f"fall through (top<thresh =&gt; visual fires); ~half of text queries skip "
        f"visual (top>=thresh =&gt; text-only)"
    )
    return threshold, note


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
    retriever, n_chunks = await _build_text_retriever(
        qdrant_url=qdrant_url,
        collection=collection,
        ollama_url=ollama_url,
        rerank_length_norm=rerank_length_norm,
        region_boost=region_boost,
    )
    golden_set = load_golden_set(golden_path)
    print(f"Calibrating cascade against {len(golden_set.queries)} queries from "
          f"{golden_set.name}/{golden_set.version}")
    print(f"Corpus: {n_chunks} chunks in '{collection}'")
    print("Mode: text-leg only (mirrors cascade's first pass).")
    print()

    records: list[dict[str, Any]] = []
    for q in golden_set.queries:
        filters: dict[str, str] = {}
        if paper_id_filter and q.paper_id:
            filters["paper_id"] = q.paper_id
        rag_query = Query(text=q.text, top_k=top_k, filters=filters)
        results = await retriever.retrieve(rag_query)
        max_score = max((r.score for r in results), default=None)
        records.append(
            {
                "query_id": q.query_id,
                "category": q.category,
                # category mode would route this query to: hybrid or text?
                "category_routes_to": "hybrid"
                if classify_query(q.text) in _HYBRID_CATEGORIES
                else "text",
                "paper_id": q.paper_id,
                "n_returned": len(results),
                "max_score": max_score,
                "top_chunk_id": results[0].chunk_id if results else None,
            }
        )

    # Per-category distribution
    by_cat: dict[str, list[float]] = {}
    for r in records:
        if r["max_score"] is None:
            continue
        by_cat.setdefault(str(r["category"]), []).append(float(r["max_score"]))

    print(f"{'query_id':40s} {'cat':14s} {'cat->':6s} {'top_score':>10s}")
    print("-" * 75)
    for r in sorted(records, key=lambda x: (x["category"], -(x["max_score"] or -999))):
        s = f"{r['max_score']:.4f}" if r["max_score"] is not None else "(none)"
        print(f"{r['query_id']:40s} {r['category']:14s} {r['category_routes_to']:6s} {s:>10s}")
    print()
    print("Per-category top-score distribution:")
    for cat, scores in sorted(by_cat.items()):
        ss = sorted(scores)
        print(
            f"  {cat:15s} n={len(ss):2d} "
            f"min={ss[0]:.3f} med={ss[len(ss)//2]:.3f} max={ss[-1]:.3f}"
        )
    print()

    threshold, note = _suggest_threshold(records)
    print(f"Suggested cascade_confidence_threshold: {threshold:.4f}")
    print(f"  ({note})")

    # Counterfactual: at this threshold, how many queries would fire visual?
    if threshold is not None:
        would_fire = [r for r in records if r["max_score"] is not None and r["max_score"] < threshold]
        wf_by_cat: dict[str, int] = {}
        for r in would_fire:
            wf_by_cat[str(r["category"])] = wf_by_cat.get(str(r["category"]), 0) + 1
        in_corpus_total = sum(1 for r in records if r["category"] != "out_of_corpus")
        text_skipped = in_corpus_total - len(
            [r for r in would_fire if r["category"] != "out_of_corpus"]
        )
        print()
        print(f"At threshold={threshold:.4f}:")
        print(f"  visual leg would fire on {len(would_fire)}/{len(records)} queries; "
              f"by category: {wf_by_cat}")
        print(f"  visual leg would SKIP on {text_skipped} in-corpus queries (cost saved)")

    if out_path:
        out_path.write_text(
            json.dumps(
                {
                    "config": {
                        "collection": collection,
                        "rerank_length_norm": rerank_length_norm,
                        "region_boost": region_boost,
                        "paper_id_filter": paper_id_filter,
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
        help="Qdrant collection holding the corpus to calibrate against.",
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
        default=Path("data/eval/runs/calibration-cascade.json"),
        help="Per-query JSON output path. Pass empty string to skip.",
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
