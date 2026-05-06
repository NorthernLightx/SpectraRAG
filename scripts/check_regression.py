"""Compare a fresh EvalRun JSON against a committed baseline. Fail if any metric
drops more than the threshold (default 5%).

CI fails if any metric drops more than 5% vs. the last main-branch baseline.
This script is the gate. It is deliberately offline — it doesn't run the
eval, only compares two JSON snapshots.

Run:
  uv run python -m scripts.check_regression \
      --baseline data/eval/baseline.json \
      --candidate data/eval/runs/run-20260501-112706.json \
      --threshold 0.05

Exit 0  — all gated metrics within threshold; prints a summary diff.
Exit 1  — at least one metric regressed beyond threshold; prints which.
Exit 2  — input file invalid or missing required fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Metrics gated by default. Each entry is (json_path_into_per_query_aggregation,
# direction). All current metrics are higher-is-better; a drop fails the gate.
_DEFAULT_METRICS = (
    "ndcg_at_5",
    "recall_at_10",
    "mrr",
    "faithfulness",
    "answer_relevance",
    "context_precision",
)


@dataclass(frozen=True)
class MetricDelta:
    name: str
    baseline: float | None
    candidate: float | None
    delta_abs: float | None
    delta_rel: float | None
    regressed: bool


# Retrieval metrics are macro-averaged over in-corpus queries only (OOC has no
# relevant chunks, so nDCG/recall/MRR are 0 by construction). Generation metrics
# are averaged over ALL queries with a non-None value — RAGAS-style: faithfulness
# of an OOC refusal is meaningful (1.0 = no hallucinated claim). Mirrors report.py.
_RETRIEVAL_FIELDS = frozenset({"ndcg_at_5", "recall_at_10", "mrr"})


def _macro_mean(per_query: list[dict[str, Any]], field: str) -> float | None:
    """Mean of `field` across queries (mirrors `src/eval/report.py` aggregation).

    - retrieval fields → in-corpus queries only
    - generation fields → all queries with a non-None value for `field`
    """
    queries = (
        [q for q in per_query if q.get("category") != "out_of_corpus"]
        if field in _RETRIEVAL_FIELDS
        else per_query
    )
    values: list[float] = []
    for q in queries:
        for container in (q.get("retrieval") or {}, q.get("generation") or {}):
            if container.get(field) is not None:
                values.append(float(container[field]))
                break
    if not values:
        return None
    return sum(values) / len(values)


def _compute_deltas(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    metrics: tuple[str, ...],
    threshold: float,
) -> list[MetricDelta]:
    deltas: list[MetricDelta] = []
    for metric in metrics:
        b = _macro_mean(baseline["per_query"], metric)
        c = _macro_mean(candidate["per_query"], metric)
        if b is None or c is None:
            deltas.append(MetricDelta(metric, b, c, None, None, regressed=False))
            continue
        delta_abs = c - b
        delta_rel = delta_abs / b if b > 0 else 0.0
        regressed = delta_rel < -threshold
        deltas.append(MetricDelta(metric, b, c, delta_abs, delta_rel, regressed))
    return deltas


def _format_table(deltas: list[MetricDelta]) -> str:
    header = (
        f"{'metric':<22}{'baseline':>12}{'candidate':>12}{'delta abs':>10}{'delta rel':>10}  status"
    )
    rows = [header, "-" * len(header)]
    for d in deltas:
        if d.baseline is None or d.candidate is None:
            rows.append(f"{d.name:<22}{'—':>12}{'—':>12}{'—':>10}{'—':>10}  not-applicable")
            continue
        b = f"{d.baseline:.4f}"
        c = f"{d.candidate:.4f}"
        da = f"{d.delta_abs:+.4f}" if d.delta_abs is not None else "—"
        dr = f"{(d.delta_rel * 100):+.2f}%" if d.delta_rel is not None else "—"
        status = "FAIL" if d.regressed else "ok"
        rows.append(f"{d.name:<22}{b:>12}{c:>12}{da:>10}{dr:>10}  {status}")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Fractional regression that fails the gate (default 0.05 = 5%%).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(_DEFAULT_METRICS),
        help="Which fields to gate. Defaults: %(default)s.",
    )
    args = parser.parse_args()

    try:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    if "per_query" not in baseline or "per_query" not in candidate:
        print("error: both files must have a 'per_query' field", file=sys.stderr)
        sys.exit(2)

    deltas = _compute_deltas(baseline, candidate, tuple(args.metrics), args.threshold)
    print(_format_table(deltas))
    print()

    regressions = [d for d in deltas if d.regressed]
    if regressions:
        print(
            f"FAIL: {len(regressions)} metric(s) regressed > {args.threshold * 100:.1f}%: "
            f"{', '.join(d.name for d in regressions)}"
        )
        sys.exit(1)
    print("PASS: no metrics regressed beyond threshold.")


if __name__ == "__main__":
    main()
