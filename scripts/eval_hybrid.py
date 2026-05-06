"""Offline RRF fusion of an existing text run + an existing visual run.

ADR 0004 names hybrid text+visual fusion as the natural follow-up to visual
retrieval. Visual recall@10 is 1.0 on golden v2 in-corpus and visual wins on
the exact queries text loses on (q4/q9/q10/q12/q20), so RRF over the two
retrieval rank lists is expected to lift nDCG@5 / MRR without re-ingestion
or re-eval.

This script consumes two existing run JSONs (no model calls, no GPU): a text
run from `scripts/eval_run.py` and a visual run from `scripts/eval_visual.py`.
Granularity is reconciled at *page level* — text chunk ids `paper::pN::cM`
are normalised to page ids `paper::pN::page` (matching visual's existing
format) and then fused via reciprocal rank fusion.

Two outputs are written so the comparison is apples-to-apples:

  * `run-text-page-<ts>.json/.md` — the *text* run re-scored at page level.
    This is the fair baseline the hybrid is compared against (the original
    text run scores at chunk level, which doesn't share an ID space with
    visual).
  * `run-hybrid-<ts>.json/.md` — the RRF-fused page rankings + metrics.
  * `run-hybrid-<ts>.compare.md` — side-by-side aggregate + per-query Δ.

Run:
  uv run python -m scripts.eval_hybrid \\
      --text-run data/eval/runs/run-20260501-190915.json \\
      --visual-run data/eval/runs/run-visual-20260501-215441.json \\
      --golden data/golden/v2.yaml
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from src.eval.golden_set import load_golden_set
from src.eval.metrics_retrieval import ndcg_at_k, recall_at_k, reciprocal_rank
from src.eval.report import write_run_json, write_run_markdown
from src.observability.logging import configure_logging, get_logger
from src.rag.hybrid import RankedItem, reciprocal_rank_fusion
from src.types import (
    EvalRun,
    GoldenQuery,
    GoldenSet,
    PerQueryResult,
    RetrievalMetrics,
)


def _chunk_id_to_page_id(chunk_id: str) -> str:
    """Normalise either text (`paper::pN::cM`) or visual (`paper::pN::page`)
    ids to canonical page form `paper::pN::page`. Idempotent on the visual
    form."""
    parts = chunk_id.split("::")
    if len(parts) < 2:
        raise ValueError(f"malformed id (need 'paper::pN::*'): {chunk_id!r}")
    paper, page = parts[0], parts[1]
    if not page.startswith("p") or not page[1:].isdigit():
        raise ValueError(f"malformed id (need 'pN' in 2nd segment): {chunk_id!r}")
    return f"{paper}::{page}::page"


def _chunks_to_pages(chunk_ids: list[str]) -> list[str]:
    """Map a ranked chunk list to a ranked unique-page list, preserving
    first-occurrence order (so the highest-ranked chunk on each page wins)."""
    seen: set[str] = set()
    pages: list[str] = []
    for cid in chunk_ids:
        page_id = _chunk_id_to_page_id(cid)
        if page_id not in seen:
            seen.add(page_id)
            pages.append(page_id)
    return pages


def _relevant_pages_for(query: GoldenQuery) -> list[str]:
    """Derive page-level ground truth: prefer explicit `relevant_pages`,
    otherwise project chunk-level labels onto pages."""
    if query.relevant_pages:
        return [f"{query.paper_id}::p{p}::page" for p in query.relevant_pages]
    if not query.relevant_chunk_ids:
        return []
    page_nums: set[int] = set()
    for cid in query.relevant_chunk_ids:
        page_id = _chunk_id_to_page_id(cid)
        page_segment = page_id.split("::")[1]
        page_nums.add(int(page_segment[1:]))
    return [f"{query.paper_id}::p{p}::page" for p in sorted(page_nums)]


def _fuse_pages(
    text_pages: list[str],
    visual_pages: list[str],
    *,
    rrf_k: int,
    top_k: int,
) -> list[str]:
    """RRF-fuse two ranked page lists. Empty lists are skipped (a single
    non-empty list is returned in original order, capped at top_k)."""
    lists: list[list[RankedItem]] = []
    if text_pages:
        lists.append([RankedItem(id=p, score=0.0) for p in text_pages])
    if visual_pages:
        lists.append([RankedItem(id=p, score=0.0) for p in visual_pages])
    if not lists:
        return []
    fused = reciprocal_rank_fusion(lists, k=rrf_k, top_k=top_k)
    return [item.id for item in fused]


def _validate_runs_compatible(text: EvalRun, visual: EvalRun) -> None:
    if text.golden_set_name != visual.golden_set_name:
        raise ValueError(
            "runs use different golden_set: "
            f"text={text.golden_set_name!r} visual={visual.golden_set_name!r}"
        )
    if text.golden_set_version != visual.golden_set_version:
        raise ValueError(
            "runs use different golden_set version: "
            f"text={text.golden_set_version!r} visual={visual.golden_set_version!r}"
        )


def _scored_per_query(
    golden: GoldenQuery,
    *,
    retrieved_pages: list[str],
    latency_ms: int,
) -> PerQueryResult:
    relevant = _relevant_pages_for(golden)
    return PerQueryResult(
        query_id=golden.query_id,
        category=golden.category,
        text=golden.text,
        retrieved_chunk_ids=retrieved_pages,
        retrieval=RetrievalMetrics(
            ndcg_at_5=ndcg_at_k(relevant, retrieved_pages, k=5),
            recall_at_10=recall_at_k(relevant, retrieved_pages, k=10),
            mrr=reciprocal_rank(relevant, retrieved_pages),
        ),
        latency_ms=latency_ms,
    )


def _hybrid_per_query(
    golden: GoldenQuery,
    *,
    text: PerQueryResult,
    visual: PerQueryResult,
    rrf_k: int,
    top_k: int,
) -> PerQueryResult:
    fused = _fuse_pages(
        _chunks_to_pages(text.retrieved_chunk_ids),
        _chunks_to_pages(visual.retrieved_chunk_ids),
        rrf_k=rrf_k,
        top_k=top_k,
    )
    return _scored_per_query(golden, retrieved_pages=fused, latency_ms=0)


def _text_page_per_query(
    golden: GoldenQuery,
    *,
    text: PerQueryResult,
    top_k: int,
) -> PerQueryResult:
    pages = _chunks_to_pages(text.retrieved_chunk_ids)[:top_k]
    return _scored_per_query(golden, retrieved_pages=pages, latency_ms=text.latency_ms)


def _aggregate(per_query: list[PerQueryResult]) -> dict[str, float]:
    in_corpus = [q for q in per_query if q.category != "out_of_corpus"]
    if not in_corpus:
        return {"ndcg5": 0.0, "recall10": 0.0, "mrr": 0.0, "n": 0.0}
    n = len(in_corpus)
    return {
        "ndcg5": sum(q.retrieval.ndcg_at_5 for q in in_corpus) / n,
        "recall10": sum(q.retrieval.recall_at_10 for q in in_corpus) / n,
        "mrr": sum(q.retrieval.mrr for q in in_corpus) / n,
        "n": float(n),
    }


def _build_runs(
    *,
    golden_set: GoldenSet,
    text_run: EvalRun,
    visual_run: EvalRun,
    rrf_k: int,
    top_k: int,
) -> tuple[EvalRun, EvalRun]:
    """Returns `(text_page_baseline, hybrid)` — both at page granularity."""
    text_by_qid = {p.query_id: p for p in text_run.per_query}
    visual_by_qid = {p.query_id: p for p in visual_run.per_query}

    text_page_results: list[PerQueryResult] = []
    hybrid_results: list[PerQueryResult] = []
    for golden in golden_set.queries:
        if golden.query_id not in text_by_qid:
            raise ValueError(f"text run missing query: {golden.query_id}")
        if golden.query_id not in visual_by_qid:
            raise ValueError(f"visual run missing query: {golden.query_id}")
        text_pq = text_by_qid[golden.query_id]
        visual_pq = visual_by_qid[golden.query_id]
        text_page_results.append(_text_page_per_query(golden, text=text_pq, top_k=top_k))
        hybrid_results.append(
            _hybrid_per_query(golden, text=text_pq, visual=visual_pq, rrf_k=rrf_k, top_k=top_k)
        )

    now = datetime.now(UTC)
    paper_ids = text_run.config.get("paper_ids", [])
    text_page_run = EvalRun(
        run_id=uuid4().hex[:12],
        started_at=now,
        finished_at=now,
        golden_set_name=golden_set.name,
        golden_set_version=golden_set.version,
        config={
            "retriever": "text-at-page",
            "source_run": text_run.run_id,
            "top_k": top_k,
            "paper_ids": paper_ids,
        },
        per_query=text_page_results,
    )
    hybrid_run = EvalRun(
        run_id=uuid4().hex[:12],
        started_at=now,
        finished_at=now,
        golden_set_name=golden_set.name,
        golden_set_version=golden_set.version,
        config={
            "retriever": "hybrid-text-visual-page",
            "fusion": "rrf",
            "rrf_k": rrf_k,
            "top_k": top_k,
            "text_source_run": text_run.run_id,
            "visual_source_run": visual_run.run_id,
            "paper_ids": paper_ids,
        },
        per_query=hybrid_results,
    )
    return text_page_run, hybrid_run


def _comparison_markdown(
    *,
    text_run_id: str,
    visual_run_id: str,
    text_page: EvalRun,
    hybrid: EvalRun,
) -> str:
    tp = _aggregate(text_page.per_query)
    hyb = _aggregate(hybrid.per_query)

    def _delta_pct(new: float, old: float) -> str:
        if old == 0:
            return "—"
        return f"{(new - old) / old * 100:+.1f}%"

    lines = [
        "# Hybrid (text + visual) — offline RRF fusion at page granularity",
        "",
        f"- **Text source run:** `{text_run_id}` (re-scored at page level)",
        f"- **Visual source run:** `{visual_run_id}`",
        f"- **Text-page run id:** `{text_page.run_id}`",
        f"- **Hybrid run id:** `{hybrid.run_id}`",
        f"- **Golden:** {hybrid.golden_set_name} {hybrid.golden_set_version}",
        f"- **n in-corpus:** {int(tp.get('n', 0))}",
        "",
        "## Aggregate (in-corpus macro)",
        "",
        "| Metric | Text @ page | Hybrid (RRF) | Δ |",
        "|---|---|---|---|",
        f"| nDCG@5 | {tp['ndcg5']:.4f} | {hyb['ndcg5']:.4f} | {_delta_pct(hyb['ndcg5'], tp['ndcg5'])} |",
        f"| recall@10 | {tp['recall10']:.4f} | {hyb['recall10']:.4f} | {_delta_pct(hyb['recall10'], tp['recall10'])} |",
        f"| MRR | {tp['mrr']:.4f} | {hyb['mrr']:.4f} | {_delta_pct(hyb['mrr'], tp['mrr'])} |",
        "",
        "## Per-query nDCG@5",
        "",
        "| query_id | category | text @ page | hybrid | Δ |",
        "|---|---|---|---|---|",
    ]
    for tp_q, hyb_q in zip(text_page.per_query, hybrid.per_query, strict=True):
        delta = hyb_q.retrieval.ndcg_at_5 - tp_q.retrieval.ndcg_at_5
        lines.append(
            f"| `{tp_q.query_id}` | {tp_q.category} | "
            f"{tp_q.retrieval.ndcg_at_5:.3f} | "
            f"{hyb_q.retrieval.ndcg_at_5:.3f} | "
            f"{delta:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-run", type=Path, required=True)
    parser.add_argument("--visual-run", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=Path("data/golden/v2.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval/runs"))
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = Path("logs") / f"eval-hybrid-{timestamp}.log"
    configure_logging(level="INFO", env="local", log_file=log_file)
    log = get_logger("scripts.eval_hybrid")
    print(f"Logging JSON to {log_file}")

    text_run = EvalRun.model_validate_json(args.text_run.read_text(encoding="utf-8"))
    visual_run = EvalRun.model_validate_json(args.visual_run.read_text(encoding="utf-8"))
    _validate_runs_compatible(text_run, visual_run)

    golden_set = load_golden_set(args.golden)
    print(
        f"Loaded golden set {golden_set.name} {golden_set.version} "
        f"({len(golden_set.queries)} queries)"
    )

    text_page_run, hybrid_run = _build_runs(
        golden_set=golden_set,
        text_run=text_run,
        visual_run=visual_run,
        rrf_k=args.rrf_k,
        top_k=args.top_k,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    text_page_json = args.output_dir / f"run-text-page-{timestamp}.json"
    text_page_md = args.output_dir / f"run-text-page-{timestamp}.md"
    hybrid_json = args.output_dir / f"run-hybrid-{timestamp}.json"
    hybrid_md = args.output_dir / f"run-hybrid-{timestamp}.md"
    compare_md = args.output_dir / f"run-hybrid-{timestamp}.compare.md"

    write_run_json(text_page_run, text_page_json)
    write_run_markdown(text_page_run, text_page_md)
    write_run_json(hybrid_run, hybrid_json)
    write_run_markdown(hybrid_run, hybrid_md)
    compare_md.write_text(
        _comparison_markdown(
            text_run_id=text_run.run_id,
            visual_run_id=visual_run.run_id,
            text_page=text_page_run,
            hybrid=hybrid_run,
        ),
        encoding="utf-8",
    )

    log.info(
        "hybrid_eval.done",
        text_run_id=text_run.run_id,
        visual_run_id=visual_run.run_id,
        text_page_run_id=text_page_run.run_id,
        hybrid_run_id=hybrid_run.run_id,
        compare_md=str(compare_md),
    )

    tp = _aggregate(text_page_run.per_query)
    hyb = _aggregate(hybrid_run.per_query)
    print(
        f"\nText @ page (in-corpus): "
        f"nDCG@5={tp['ndcg5']:.4f} recall@10={tp['recall10']:.4f} MRR={tp['mrr']:.4f}"
    )
    print(
        f"Hybrid       (in-corpus): "
        f"nDCG@5={hyb['ndcg5']:.4f} recall@10={hyb['recall10']:.4f} MRR={hyb['mrr']:.4f}"
    )
    print(f"\nWrote {text_page_json}")
    print(f"Wrote {text_page_md}")
    print(f"Wrote {hybrid_json}")
    print(f"Wrote {hybrid_md}")
    print(f"Wrote {compare_md}")


if __name__ == "__main__":
    main()
