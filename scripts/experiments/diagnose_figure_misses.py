"""Decompose figure-subset retrieval misses into ranking-loss vs true-miss.

THE FORK THIS ANSWERS (strategy session, 2026-05-24): of the figure gold
PAGES missed at rank k_recall by the shipped router, how many are

  (a) ranked past k_recall but retrieved by depth k_coverage
      -> fix = fusion / rerank / top-K, CHEAP
  (b) never retrieved by any leg at depth k_coverage
      -> true-miss, fix = better visual retriever / higher DPI, EXPENSIVE

(a) splits further: a missed page inside the FUSED top-k_coverage is plain
ranking-loss; one that a single leg retrieved but fusion dropped below
k_coverage is "fusion-buried". Both are the cheap lever, but the split says
whether to touch the reranker (ranking-loss) or the fusion weighting
(fusion-buried).

Routing is a closed lever (ADR 0013: shipped router == oracle on this
benchmark), so the remaining figure headroom is one of the above. The
committed top-10-only baselines cannot tell them apart; this consumes a
*depth-50* run from diagnose_depth50_run.py (each leg's top-50 + fused).

The decomposition is GOLD-PAGE-LEVEL (micro): it partitions every gold page,
so a query with 2 gold pages that finds 1 contributes one hit + one miss --
the same accounting rescore_mmlb_pages.py uses for fractional recall. 44% of
figure queries here carry >1 gold page, so a query-binary view would discard
nearly half the missed-page signal. The macro recall@k_recall is printed too
and must reproduce the committed router figure (0.7578) as a tie-in check.

Page logic mirrors scripts/rescore_mmlb_pages.py EXACTLY (paper-aware
`paper::pN::cM` / `paper::pN::page` -> (paper, N), dedup-in-rank).

Usage:

    .venv/Scripts/python.exe -m scripts.experiments.diagnose_figure_misses \
        --run data/eval/runs/depth50-<ts>/depth50.json \
        --golden data/golden/mmlongbench-v1.yaml \
        --k-recall 10 --k-coverage 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# Identical to scripts/rescore_mmlb_pages.py -- a page is (paper_id, page_no).
_PAGE_RE = re.compile(r"::p(\d+)")
Page = tuple[str, int]
# One scored query: (query_id, relevant pages, fused / text / visual rankings).
Row = tuple[str, set[Page], list[Page], list[Page], list[Page]]


def _page_of(chunk_id: str) -> Page | None:
    match = _PAGE_RE.search(chunk_id)
    if match is None:
        return None
    return chunk_id.split("::", 1)[0], int(match.group(1))


def _pages_in_rank(chunk_ids: list[str]) -> list[Page]:
    """Walk chunk-ids in rank order; keep each page once at first appearance."""
    seen: set[Page] = set()
    pages: list[Page] = []
    for chunk_id in chunk_ids:
        page = _page_of(chunk_id)
        if page is None or page in seen:
            continue
        seen.add(page)
        pages.append(page)
    return pages


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="depth-50 run JSON")
    parser.add_argument("--golden", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    parser.add_argument("--k-recall", type=int, default=10, help="rank cutoff for the gated metric")
    parser.add_argument("--k-coverage", type=int, default=50, help="depth at which to measure coverage")
    parser.add_argument(
        "--category",
        default="figure",
        help="subset to decompose (figure / table / factual). Default figure.",
    )
    args = parser.parse_args()

    run = json.loads(args.run.read_text(encoding="utf-8"))
    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))

    relevant_by_qid: dict[str, set[Page]] = {
        q["query_id"]: {
            (q["paper_id"], page)
            for page in (q.get("relevant_pages") or [])
            if q.get("paper_id")
        }
        for q in golden["queries"]
    }

    kr, kc = args.k_recall, args.k_coverage

    # Each per_query record must carry the three rank lists the depth-50 run
    # persists: text leg, visual leg, fused router output (all top-50).
    queries: list[Row] = []
    missing_fields = 0
    for pq in run.get("per_query", []):
        if pq.get("category") != args.category:
            continue
        relevant = relevant_by_qid.get(pq["query_id"])
        if not relevant:  # label-gap or OOC -- dropped, same as rescore
            continue
        if not all(key in pq for key in ("text_top50", "visual_top50", "fused_top50")):
            missing_fields += 1
            continue
        queries.append(
            (
                pq["query_id"],
                relevant,
                _pages_in_rank(pq["fused_top50"]),
                _pages_in_rank(pq["text_top50"]),
                _pages_in_rank(pq["visual_top50"]),
            )
        )

    if missing_fields:
        print(
            f"WARNING: {missing_fields} {args.category} queries lacked "
            "text_top50/visual_top50/fused_top50 -- run was not produced by "
            "diagnose_depth50_run.py. Decomposition is incomplete.\n"
        )
    if not queries:
        print(f"No scored {args.category} queries with depth-50 leg data. Nothing to decompose.")
        sys.exit(2)

    # --- macro fractional metrics (tie back to the committed headline) ---
    def frac_recall(fused: list[Page], rel: set[Page], k: int) -> float:
        return sum(1 for p in fused[:k] if p in rel) / len(rel)

    macro_recall = _avg([frac_recall(f, rel, kr) for _, rel, f, _, _ in queries])
    macro_cov = _avg([frac_recall(f, rel, kc) for _, rel, f, _, _ in queries])

    # --- gold-page-level (micro) decomposition ---
    hit = ranking_loss = fusion_buried = retriever_miss = 0
    audit: list[tuple[str, Page, str]] = []
    for qid, rel, fused, text, vis in queries:
        fused_kr = set(fused[:kr])
        fused_kc = set(fused[:kc])
        legs_kc = set(text[:kc]) | set(vis[:kc])
        for page in sorted(rel):
            if page in fused_kr:
                hit += 1
            elif page in fused_kc:
                ranking_loss += 1
                audit.append((qid, page, "ranking-loss"))
            elif page in legs_kc:
                fusion_buried += 1
                audit.append((qid, page, "fusion-buried"))
            else:
                retriever_miss += 1
                audit.append((qid, page, "retriever-miss"))

    total_gold = hit + ranking_loss + fusion_buried + retriever_miss
    missed = ranking_loss + fusion_buried + retriever_miss
    cheap = ranking_loss + fusion_buried

    print(
        f"Figure-miss decomposition  (subset={args.category!r}, "
        f"n={len(queries)} queries, {total_gold} gold pages)"
    )
    print(f"  k_recall={kr}  k_coverage={kc}\n")
    print(f"  macro recall@{kr}    (fused, ties to committed 0.7578) : {macro_recall:.4f}")
    print(f"  macro coverage@{kc}  (fused)                           : {macro_cov:.4f}")
    print(f"  gold pages: {total_gold} total | {hit} hit@{kr} | {missed} missed\n")

    if not missed:
        print("  No missed gold pages -- nothing to decompose.")
        return

    print("  --- THE FORK (of the missed gold pages) ---")
    print(f"  CHEAP fix     (fusion / rerank / top-K) : {cheap}/{missed}  ({cheap / missed:.1%})")
    print(f"    ranking-loss  (in fused top-{kc}, past rank {kr}) : {ranking_loss}")
    print(f"    fusion-buried (a leg had it, fusion dropped it)  : {fusion_buried}")
    print(
        f"  EXPENSIVE fix (better retriever / DPI)  : "
        f"{retriever_miss}/{missed}  ({retriever_miss / missed:.1%})"
    )
    print(f"    true-miss     (no leg retrieved it at depth {kc}) : {retriever_miss}\n")

    # Per-page audit trail -- proves the split isn't an aggregation artefact.
    print(f"  missed gold pages ({missed}):")
    print(f"  {'query_id':<40} {'page':>18} {'verdict':>16}")
    for qid, (paper, pno), verdict in audit:
        label = f"{paper[:12]}::p{pno}"
        print(f"  {qid:<40} {label:>18} {verdict:>16}")


if __name__ == "__main__":
    main()
