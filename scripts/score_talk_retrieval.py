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

from scripts.derive_arms import read_run
from src.eval.golden_set import load_golden_set
from src.eval.metrics_retrieval import recall_at_k, reciprocal_rank
from src.eval.spans import (
    pages_for_spans,
    random_hit_at_k,
    random_recall_at_k,
    random_reciprocal_rank,
)
from src.ingestion.media import load_manifest
from src.types import MediaSegment
from src.types.eval import GoldenQuery

_METRICS = ("hit_at_1", "hit_at_3", "recall_at_1", "recall_at_3", "mrr")
_BOOTSTRAP = 2000


def _page(chunk_id: str) -> str:
    parts = chunk_id.split("::")
    return "::".join(parts[:2]) if len(parts) >= 2 else chunk_id


def _page_number(page_id: str) -> int:
    return int(page_id.rsplit("::p", 1)[1])


def _metrics(relevant: Sequence[str], ranked: Sequence[str]) -> dict[str, float]:
    return {
        # A hit counts evidence spread over several pages in full, where
        # recall@1 can reach only 1/len(pages).
        "hit_at_1": float(bool(set(relevant) & set(ranked[:1]))),
        "hit_at_3": float(bool(set(relevant) & set(ranked[:3]))),
        "recall_at_1": recall_at_k(relevant, list(ranked), k=1),
        "recall_at_3": recall_at_k(relevant, list(ranked), k=3),
        "mrr": reciprocal_rank(relevant, list(ranked)),
    }


def score_query(
    q: GoldenQuery, segments: Sequence[MediaSegment], ranked_ids: Sequence[str]
) -> dict[str, Any] | None:
    """Page-level scores of one ranking beside two references at the same
    depth: a random ranking of the recording's pages, and its pages in time
    order (which title-slide questions reward). An arm cut to its top k is
    compared with k pages, not with the whole recording.

    None when the query's spans lie outside every page, which no ranking can
    find. Raises when the ranking holds another recording's pages (a run made
    without --paper-id-filter) or a page the manifest lacks (a run and a
    manifest from different segmentations): either would skew every score."""
    foreign = next((c for c in ranked_ids if c.split("::")[0] != q.paper_id), None)
    if foreign is not None:
        raise ValueError(
            f"{q.query_id}: the ranking holds other recordings' pages ({foreign}); "
            "score runs made with --paper-id-filter"
        )
    n = len(segments)
    ranked = list(dict.fromkeys(_page(c) for c in ranked_ids))
    beyond = next((page for page in ranked if _page_number(page) > n), None)
    if beyond is not None:
        raise ValueError(
            f"{q.query_id}: ranked page {beyond} is not in the manifest ({n} segments); "
            "the run and the manifests come from different segmentations"
        )
    pages = pages_for_spans(q.relevant_spans, segments)
    if not pages:
        return None
    relevant = [f"{q.paper_id}::p{p}" for p in pages]
    depth = min(len(ranked), n)
    in_order = [f"{q.paper_id}::p{p}" for p in range(1, depth + 1)]
    row: dict[str, Any] = {
        "query_id": q.query_id,
        "category": q.category,
        "origin": _origin(q.note),
        "width": _width(len(pages)),
        "relevant_pages": pages,
        "n_pages": n,
        "depth": depth,
        **_metrics(relevant, ranked),
        **{f"in_order_{m}": v for m, v in _metrics(relevant, in_order).items()},
        "random_hit_at_1": random_hit_at_k(n_pages=n, n_relevant=len(pages), k=min(1, depth)),
        "random_hit_at_3": random_hit_at_k(n_pages=n, n_relevant=len(pages), k=min(3, depth)),
        "random_recall_at_1": random_recall_at_k(n_pages=n, k=min(1, depth)),
        "random_recall_at_3": random_recall_at_k(n_pages=n, k=min(3, depth)),
        "random_mrr": random_reciprocal_rank(n_pages=n, n_relevant=len(pages), depth=depth),
    }
    return row


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
            "in_order": sum(r[f"in_order_{m}"] for r in rows) / len(rows),
            "gain_ci95": [low, high],
        }
    return out


def _print(label: str, summary: dict[str, Any]) -> None:
    cells = [
        f"{m} {s['mean']:.3f} (random {s['random']:.3f}, in order {s['in_order']:.3f}, "
        f"gain on random {s['gain_ci95'][0]:+.3f}"
        f"..{s['gain_ci95'][1]:+.3f})"
        for m, s in ((m, summary[m]) for m in _METRICS)
    ]
    print(f"{label:<22} n={summary['n']:<4} " + "  ".join(cells))


def score_run(
    run: dict[str, Any],
    golden: dict[str, GoldenQuery],
    pages_dir: Path,
    leg: str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Per-query rows for every span-labelled query of `golden` in `run`, and
    the ids of those whose spans map to no page."""
    segments: dict[str, list[MediaSegment]] = {}
    rows: list[dict[str, Any]] = []
    unmapped: list[str] = []
    for pq in run["per_query"]:
        q = golden.get(pq["query_id"])
        if q is None:
            continue
        if q.paper_id not in segments:
            segments[q.paper_id] = load_manifest(pages_dir, q.paper_id).segments
        ranked = (pq.get("leg_chunk_ids") or {}).get(leg) if leg else pq["retrieved_chunk_ids"]
        if ranked is None:
            raise SystemExit(f"{q.query_id}: the run recorded no {leg} leg (use --router)")
        row = score_query(q, segments[q.paper_id], ranked)
        if row is None:
            unmapped.append(q.query_id)
        else:
            rows.append(row)
    return rows, unmapped


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
        run = read_run(path)
        rows, unmapped = score_run(run, golden, args.pages_dir, args.leg)
        missing = sorted(set(golden) - {pq["query_id"] for pq in run["per_query"]})
        if not rows:
            raise SystemExit(f"{path.name}: no span-labelled query of the golden set is in it")
        groups: dict[str, list[dict[str, Any]]] = {"all": rows}
        for key in ("width", "category", "origin"):
            for r in rows:
                groups.setdefault(f"{key}={r[key]}", []).append(r)
        summaries = {name: summarize(g) for name, g in groups.items()}
        source = f"{args.leg} leg" if args.leg else "retrieved ranking"
        print(f"{path.name} | {source} | {len(rows)} labelled queries")
        if unmapped or missing:
            print(f"  not scored: {len(unmapped)} with no page, {len(missing)} absent from the run")
        for name, summary in summaries.items():
            _print(name, summary)
        if args.output:
            args.output.write_text(
                json.dumps({"summary": summaries, "per_query": rows}, indent=1), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
