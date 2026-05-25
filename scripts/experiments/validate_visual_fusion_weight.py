"""Re-fuse the depth-50 per-leg run through the real ADR 0023 weighted RRF.

CPU-only validation for the visual-favoring fusion weight: feeds the existing
`text_top50` / `visual_top50` legs of a depth-50 run JSON through the *actual*
`RoutingRetriever._fuse_page_level` at a sweep of visual weights, then scores the
re-fused ranking with the canonical page-level helpers from
`scripts.rescore_mmlb_pages`. No model inference — it only re-orders chunk ids
that were already retrieved.

Expected anchors (figure subset, n=75):
  w=1.0 -> recall@10 == 0.7293  (reproduces the shipped equal-weight fusion)
  w=5.0 -> recall@10 ~= 0.807   (the overnight finding)

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.validate_visual_fusion_weight \
        --run data/eval/runs/depth50-20260525-015216/depth50.json \
        --golden data/golden/mmlongbench-v1.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.rescore_mmlb_pages import (
    Page,
    _avg,
    _dedup_pages_in_rank,
    _mrr,
    _ndcg_at_k,
    _recall_at_k,
)
from src.rag.retrievers.routing import RoutingRetriever
from src.types import Query, RetrievalResult

WEIGHTS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
HYBRID_CATEGORIES = ("figure", "table", "multi_hop")


def _legs_to_results(chunk_ids: list[str], source: str) -> list[RetrievalResult]:
    """Wrap a rank-ordered chunk-id list as RetrievalResults.

    Score descends with rank so `_fuse_page_level`'s best-text-per-page pick
    lands on the first-appearing chunk of each page — the same page identity the
    rescorer keys on. The paper_id/page_numbers are parsed from the id; their
    exact values don't matter to fusion (it keys on the `paper::pN` prefix) but
    keep the RetrievalResult well-formed.
    """
    out: list[RetrievalResult] = []
    n = len(chunk_ids)
    for rank, cid in enumerate(chunk_ids):
        parts = cid.split("::")
        paper = parts[0]
        page = int(parts[1][1:]) if len(parts) > 1 and parts[1].startswith("p") else 0
        out.append(
            RetrievalResult(
                chunk_id=cid,
                paper_id=paper,
                score=float(n - rank),
                text="",
                page_numbers=[page],
                source=source,
            )
        )
    return out


async def _refuse_one(
    router: RoutingRetriever, text_ids: list[str], visual_ids: list[str], top_k: int
) -> list[str]:
    """Run the REAL _fuse_page_level and return the re-fused chunk-id order."""
    text_results = _legs_to_results(text_ids, "pipeline")
    visual_results = _legs_to_results(visual_ids, "visual")
    fused = router._fuse_page_level(text_results, visual_results, top_k=top_k)
    return [r.chunk_id for r in fused]


def _score_subset(
    per_query: list[dict[str, Any]],
    relevant_by_qid: dict[str, set[Page]],
    refused_by_qid: dict[str, list[str]],
    category: str | None,
) -> dict[str, float | int]:
    ndcg: list[float] = []
    recall: list[float] = []
    mrr: list[float] = []
    for pq in per_query:
        if category is not None and pq.get("category") != category:
            continue
        rel = relevant_by_qid.get(pq["query_id"], set())
        if not rel:
            continue
        pages = _dedup_pages_in_rank(refused_by_qid[pq["query_id"]])
        ndcg.append(_ndcg_at_k(pages, rel, k=5))
        recall.append(_recall_at_k(pages, rel, k=10))
        mrr.append(_mrr(pages, rel))
    return {
        "n": len(recall),
        "ndcg_at_5": _avg(ndcg),
        "recall_at_10": _avg(recall),
        "mrr": _avg(mrr),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    run = json.loads(args.run.read_text(encoding="utf-8"))
    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    per_query: list[dict[str, Any]] = run["per_query"]

    relevant_by_qid: dict[str, set[Page]] = {
        q["query_id"]: {
            (q["paper_id"], p) for p in (q.get("relevant_pages") or []) if q.get("paper_id")
        }
        for q in golden["queries"]
    }

    print(f"Re-fusing {len(per_query)} queries from {args.run.name} at top_k={args.top_k}")
    print("(figure subset is the validation anchor; hybrid = figure+table+multi_hop)\n")

    for w in WEIGHTS:
        # A real RoutingRetriever; we exercise only its _fuse_page_level.
        router = RoutingRetriever(
            text=_NoopRetriever(),
            visual=_NoopRetriever(),
            visual_fusion_weight=w,
        )
        refused_by_qid: dict[str, list[str]] = {}
        for pq in per_query:
            refused_by_qid[pq["query_id"]] = await _refuse_one(
                router, pq["text_top50"], pq["visual_top50"], args.top_k
            )

        fig = _score_subset(per_query, relevant_by_qid, refused_by_qid, "figure")
        tab = _score_subset(per_query, relevant_by_qid, refused_by_qid, "table")
        print(
            f"w_visual={w:>4}  | figure (n={fig['n']}): "
            f"recall@10={fig['recall_at_10']:.4f} nDCG@5={fig['ndcg_at_5']:.4f} "
            f"MRR={fig['mrr']:.4f}  | table (n={tab['n']}): "
            f"recall@10={tab['recall_at_10']:.4f} nDCG@5={tab['ndcg_at_5']:.4f}"
        )

    # Per-subset breakdown across all hybrid categories at the two anchor weights.
    print("\nFull hybrid-category breakdown at anchor weights:")
    for w in (1.0, 5.0):
        router = RoutingRetriever(
            text=_NoopRetriever(), visual=_NoopRetriever(), visual_fusion_weight=w
        )
        refused_by_qid = {
            pq["query_id"]: await _refuse_one(
                router, pq["text_top50"], pq["visual_top50"], args.top_k
            )
            for pq in per_query
        }
        print(f"  w_visual={w}")
        for cat in HYBRID_CATEGORIES:
            s = _score_subset(per_query, relevant_by_qid, refused_by_qid, cat)
            print(
                f"    {cat:<10} n={s['n']:>3}  recall@10={s['recall_at_10']:.4f} "
                f"nDCG@5={s['ndcg_at_5']:.4f} MRR={s['mrr']:.4f}"
            )


class _NoopRetriever:
    """Satisfies the Retriever protocol; never called (we call _fuse_page_level directly)."""

    async def retrieve(self, query: Query) -> list[RetrievalResult]:  # pragma: no cover
        return []


if __name__ == "__main__":
    asyncio.run(main())
