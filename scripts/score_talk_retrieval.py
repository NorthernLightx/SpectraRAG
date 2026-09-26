"""Score talk retrieval in a run JSON against time-span labels.

Relevant pages come from each query's `relevant_spans` and its talk's current
segment manifest (src/eval/spans.py), so a re-segmentation re-scores without
touching a label. A talk has only tens of segments, which makes recall@10 close
to free; this reports recall@1, recall@3 and MRR, each beside what a random
ranking of the same talk would score, with paired bootstrap intervals on the
gain over random.

Usage:
    uv run python -m scripts.score_talk_retrieval --run data/eval/runs/run-<ts>.json \\
        --golden data/golden/mcif-v1.yaml --pages-dir data/mcif/pages [--leg text]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.eval.golden_set import load_golden_set
from src.eval.metrics_retrieval import recall_at_k, reciprocal_rank
from src.eval.spans import pages_for_spans, random_recall_at_k, random_reciprocal_rank
from src.ingestion.media import load_manifest
from src.types import MediaSegment
from src.types.eval import GoldenQuery

_METRICS = ("recall_at_1", "recall_at_3", "mrr")
_BOOTSTRAP = 2000


def _page(chunk_id: str) -> str:
    parts = chunk_id.split("::")
    return "::".join(parts[:2]) if len(parts) >= 2 else chunk_id


def score_query(
    q: GoldenQuery, segments: Sequence[MediaSegment], ranked_ids: Sequence[str]
) -> dict[str, Any]:
    """Page-level scores of one ranking, and what a random ranking of the same
    talk would score at the same depth: an arm cut to its top k is compared with
    a random list of k pages, not a random ordering of the whole talk."""
    pages = pages_for_spans(q.relevant_spans, segments)
    relevant = [f"{q.paper_id}::p{n}" for n in pages]
    ranked = list(dict.fromkeys(_page(c) for c in ranked_ids))
    n = len(segments)
    depth = min(len(ranked), n)
    return {
        "query_id": q.query_id,
        "category": q.category,
        "origin": _origin(q.note),
        "width": _width(len(pages)),
        "relevant_pages": pages,
        "n_pages": n,
        "depth": depth,
        "recall_at_1": recall_at_k(relevant, ranked, k=1),
        "recall_at_3": recall_at_k(relevant, ranked, k=3),
        "mrr": reciprocal_rank(relevant, ranked),
        "random_recall_at_1": random_recall_at_k(n_pages=n, k=min(1, depth)),
        "random_recall_at_3": random_recall_at_k(n_pages=n, k=min(3, depth)),
        "random_mrr": random_reciprocal_rank(n_pages=n, n_relevant=max(1, len(pages)), depth=depth),
    }


def _width(n_pages: int) -> str:
    """Evidence width, the main driver of the random baseline."""
    if n_pages <= 1:
        return "1 page"
    return "2-3 pages" if n_pages <= 3 else "4+ pages"


def _origin(note: str | None) -> str:
    match = re.search(r"origin=(\w+)", note or "")
    return match.group(1) if match else "?"


def _interval(values: Sequence[float], rng: random.Random) -> tuple[float, float]:
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(_BOOTSTRAP))
    return means[int(0.025 * _BOOTSTRAP)], means[int(0.975 * _BOOTSTRAP) - 1]


def summarize(rows: Sequence[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    rng = random.Random(seed)
    out: dict[str, Any] = {"n": len(rows)}
    for m in _METRICS:
        gains = [r[m] - r[f"random_{m}"] for r in rows]
        low, high = _interval(gains, rng)
        out[m] = {
            "mean": sum(r[m] for r in rows) / len(rows),
            "random": sum(r[f"random_{m}"] for r in rows) / len(rows),
            "gain_ci95": [low, high],
        }
    return out


def _print(label: str, summary: dict[str, Any]) -> None:
    cells = [
        f"{m} {s['mean']:.3f} (random {s['random']:.3f}, gain {s['gain_ci95'][0]:+.3f}"
        f"..{s['gain_ci95'][1]:+.3f})"
        for m, s in ((m, summary[m]) for m in _METRICS)
    ]
    print(f"{label:<22} n={summary['n']:<4} " + "  ".join(cells))


def score_run(
    run: dict[str, Any],
    golden: dict[str, GoldenQuery],
    pages_dir: Path,
    leg: str | None,
) -> list[dict[str, Any]]:
    """Per-query rows for every span-labelled query of `golden` in `run`."""
    segments: dict[str, list[MediaSegment]] = {}
    rows: list[dict[str, Any]] = []
    for pq in run["per_query"]:
        q = golden.get(pq["query_id"])
        if q is None:
            continue
        if q.paper_id not in segments:
            segments[q.paper_id] = load_manifest(pages_dir, q.paper_id).segments
        ranked = (pq.get("leg_chunk_ids") or {}).get(leg) if leg else pq["retrieved_chunk_ids"]
        if ranked is None:
            raise SystemExit(f"{q.query_id}: the run recorded no {leg} leg (use --router)")
        rows.append(score_query(q, segments[q.paper_id], ranked))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--golden", type=Path, default=Path("data/golden/mcif-v1.yaml"))
    parser.add_argument("--pages-dir", type=Path, default=Path("data/mcif/pages"))
    parser.add_argument(
        "--leg",
        choices=("text", "visual"),
        help="score one recorded leg instead of the run's retrieved ranking",
    )
    parser.add_argument("--output", type=Path, help="write per-query rows and the summary")
    args = parser.parse_args()
    if args.output and len(args.run) > 1:
        parser.error("--output takes a single --run")

    golden = {q.query_id: q for q in load_golden_set(args.golden).queries if q.relevant_spans}
    for path in args.run:
        rows = score_run(
            json.loads(path.read_text(encoding="utf-8")), golden, args.pages_dir, args.leg
        )
        if not rows:
            raise SystemExit(f"{path.name}: no span-labelled query of the golden set is in it")
        groups: dict[str, list[dict[str, Any]]] = {"all": rows}
        for key in ("width", "category", "origin"):
            for r in rows:
                groups.setdefault(f"{key}={r[key]}", []).append(r)
        summaries = {name: summarize(g) for name, g in groups.items()}
        source = f"{args.leg} leg" if args.leg else "retrieved ranking"
        print(f"{path.name} | {source} | {len(rows)} labelled queries")
        for name, summary in summaries.items():
            _print(name, summary)
        if args.output:
            args.output.write_text(
                json.dumps({"summary": summaries, "per_query": rows}, indent=1), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
