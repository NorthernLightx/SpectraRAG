"""Recompute retrieval metrics for an existing EvalRun under updated goldens.

When goldens get updated (e.g. ADR 0009 follow-up adding region chunks to
`relevant_chunk_ids`), historical runs become unfair-comparison baselines —
their nDCG was computed against the *old* relevant set. Re-running the eval
on the updated goldens is GPU-expensive (~60 min for v3 + router + visual);
nDCG / recall / MRR are deterministic functions of (retrieved_ranks,
relevant_set), so we can recompute them offline in seconds.

Usage:
    uv run python -m scripts.rebaseline_offline \\
        --run data/eval/runs/run-20260509-002218.json \\
        --golden data/golden/v3.yaml \\
        --out data/eval/runs/run-20260509-002218.rebaselined.json

Generation metrics (faithfulness, answer_relevance, context_precision) are
copied through unchanged — they're a function of (answer, retrieved_chunks),
neither of which changes. citation_grounding likewise.

Out: a new EvalRun JSON with the same per_query / config but recomputed
retrieval metrics + an `rebaselined_against` field naming the golden file.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from src.eval.metrics_retrieval import ndcg_at_k, recall_at_k, reciprocal_rank


def _load_relevant_by_query(golden_path: Path) -> dict[str, list[str]]:
    data = yaml.safe_load(golden_path.read_text(encoding="utf-8"))
    return {q["query_id"]: list(q.get("relevant_chunk_ids") or []) for q in data["queries"]}


def _load_categories_by_query(golden_path: Path) -> dict[str, str]:
    data = yaml.safe_load(golden_path.read_text(encoding="utf-8"))
    return {q["query_id"]: q["category"] for q in data["queries"]}


def _macro_mean(per_query: list[dict[str, Any]], field: str, in_corpus_only: bool) -> float | None:
    values: list[float] = []
    for q in per_query:
        if in_corpus_only and q.get("category") == "out_of_corpus":
            continue
        for container in (q.get("retrieval") or {}, q.get("generation") or {}):
            if container.get(field) is not None:
                values.append(float(container[field]))
                break
    if not values:
        return None
    return sum(values) / len(values)


def rebaseline(run_path: Path, golden_path: Path, out_path: Path) -> dict[str, Any]:
    """Read the run JSON, recompute retrieval metrics against `golden_path`, write `out_path`.

    Returns the rebaselined run dict (also written to disk).
    """
    run = json.loads(run_path.read_text(encoding="utf-8"))
    relevant_by_q = _load_relevant_by_query(golden_path)
    cats_by_q = _load_categories_by_query(golden_path)

    new_run = copy.deepcopy(run)
    for pq in new_run["per_query"]:
        qid = pq["query_id"]
        relevant = relevant_by_q.get(qid)
        if relevant is None:
            # Query no longer in goldens — keep retrieval metrics as-is and flag.
            pq.setdefault("rebaseline_note", "query absent in updated goldens; metrics unchanged")
            continue
        retrieved = pq.get("retrieved_chunk_ids") or []
        # OOC queries are 0 by construction; keep zeros.
        if cats_by_q.get(qid) == "out_of_corpus":
            pq["retrieval"] = {"ndcg_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0}
            continue
        pq["retrieval"] = {
            "ndcg_at_5": ndcg_at_k(relevant, retrieved, k=5),
            "recall_at_10": recall_at_k(relevant, retrieved, k=10),
            "mrr": reciprocal_rank(relevant, retrieved),
        }

    new_run["rebaselined_against"] = str(golden_path)
    new_run["rebaselined_from_run_id"] = run.get("run_id")
    out_path.write_text(json.dumps(new_run, indent=2), encoding="utf-8")
    return dict(new_run)


def _print_summary(run: dict[str, Any]) -> None:
    """Aggregate retrieval + generation across per_query and print a one-line summary."""
    per_query = run["per_query"]
    rows = [
        ("nDCG@5 (in-corpus)", _macro_mean(per_query, "ndcg_at_5", in_corpus_only=True)),
        ("recall@10 (in-corpus)", _macro_mean(per_query, "recall_at_10", in_corpus_only=True)),
        ("MRR (in-corpus)", _macro_mean(per_query, "mrr", in_corpus_only=True)),
        ("faithfulness (all)", _macro_mean(per_query, "faithfulness", in_corpus_only=False)),
        (
            "answer_relevance (all)",
            _macro_mean(per_query, "answer_relevance", in_corpus_only=False),
        ),
        (
            "context_precision (all)",
            _macro_mean(per_query, "context_precision", in_corpus_only=False),
        ),
    ]
    print(f"Rebaselined run_id={run.get('run_id')} against {run.get('rebaselined_against')}")
    for label, value in rows:
        if value is None:
            print(f"  {label:25s} (n/a)")
        else:
            print(f"  {label:25s} {value:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute an EvalRun under updated goldens.")
    parser.add_argument("--run", type=Path, required=True, help="Existing run JSON to rebaseline.")
    parser.add_argument("--golden", type=Path, required=True, help="Updated golden YAML.")
    parser.add_argument("--out", type=Path, required=True, help="Output rebaselined run JSON.")
    args = parser.parse_args()
    if not args.run.exists():
        print(f"run not found: {args.run}", file=sys.stderr)
        return 2
    if not args.golden.exists():
        print(f"golden not found: {args.golden}", file=sys.stderr)
        return 2
    run = rebaseline(args.run, args.golden, args.out)
    _print_summary(run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
