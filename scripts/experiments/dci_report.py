"""Aggregate DCI run caches into a comparison vs the published BRIGHT bars.

Reads the per-query rankings each dci_eval run cached, recomputes nDCG@10 against
the gold set, and prints each method's macro mean — overall and on the subset of
queries common to all runs (a fair head-to-head when a cloud run covers fewer
queries than the local run). Published biology bars are shown for context.

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.dci_report \
        --corpus-dir data/dci/bright_biology \
        --runs gemma3:data/eval/runs/dci_bio_gemma.json \
               qwen3-235b:data/eval/runs/dci_bio_cloud.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_BARS = [
    ("BM25 (Lucene, published)", 18.9),
    ("BM25 (naive rank_bm25, ours)", 8.4),
    ("dense (best, published)", 30.0),
    ("ReasonRank-32B (published)", 58.2),
    ("DCI-Lite / GPT-5.4-nano (published)", 60.0),
    ("DCI-CC / Sonnet-4.6 (published)", 77.1),
]


def _ndcg_at_k(ranked: list[str], gold: set[str], excluded: set[str], k: int = 10) -> float:
    ranked = [d for d in ranked if d not in excluded][:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/dci/bright_biology"))
    ap.add_argument("--runs", nargs="+", required=True, help="label:path.json pairs")
    args = ap.parse_args()

    queries = {q["qid"]: q for q in (
        json.loads(line) for line in (args.corpus_dir / "queries.jsonl").open(encoding="utf-8")
    )}

    runs: dict[str, dict[str, Any]] = {}
    for spec in args.runs:
        label, _, path = spec.partition(":")
        p = Path(path)
        if not p.exists():
            print(f"  (skip {label}: {p} not found)")
            continue
        runs[label] = json.loads(p.read_text(encoding="utf-8")).get("per_query", {})

    per_run_scores: dict[str, dict[str, float]] = {}
    for label, pq in runs.items():
        scored: dict[str, float] = {}
        for qid, rec in pq.items():
            if qid not in queries:
                continue
            q = queries[qid]
            scored[qid] = _ndcg_at_k(rec["ranked"], set(q["gold_ids"]), set(q["excluded_ids"]))
        per_run_scores[label] = scored

    common = set.intersection(*(set(s) for s in per_run_scores.values())) if per_run_scores else set()

    print("=" * 64)
    print("DCI on BRIGHT-Biology — nDCG@10 (higher is better)")
    print("=" * 64)
    print("\nOurs (agentic DCI, read+grep+rank over raw corpus):")
    for label, scored in per_run_scores.items():
        full = 100 * sum(scored.values()) / len(scored) if scored else 0.0
        comm = 100 * sum(scored[q] for q in common) / len(common) if common else 0.0
        print(f"  {label:28} {full:5.1f}  (n={len(scored):3})   common-subset {comm:5.1f} (n={len(common)})")
    print("\nPublished bars (same corpus / gold_ids / nDCG@10):")
    for name, val in _BARS:
        print(f"  {name:38} {val:5.1f}")
    print("\nNote: common-subset is the fair head-to-head when runs cover different query counts.")


if __name__ == "__main__":
    main()
