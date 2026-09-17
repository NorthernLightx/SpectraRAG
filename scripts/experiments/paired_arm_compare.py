"""Paired per-query comparison of two committed eval arms.

ADR 0032 measured four arms and chose always-hybrid, but it compared each arm
to the classifier router rather than comparing the two leading arms to each
other. Visual-only scored 0.800 against hybrid's 0.784 on separate means with
overlapping intervals, which settles nothing: the arms run the same queries, so
the question is per-query and the unpaired intervals are the wrong instrument.

Pairs the arms by query_id, reports the mean delta with a paired bootstrap
interval, and runs an exact two-sided sign test over the queries where the arms
disagree. Reads committed run JSONs only. No model calls, no corpus.

Run:
  .venv\\Scripts\\python.exe -m scripts.experiments.paired_arm_compare \\
      --a data/eval/baseline-mmdocir-visual.json \\
      --b data/eval/baseline-mmdocir-hybrid.json \\
      --metric recall_at_10
"""

from __future__ import annotations

import argparse
import json
import random
from math import comb
from pathlib import Path
from typing import Any

_BOOTSTRAP_ROUNDS = 10_000
# Fixed so a rerun reproduces the interval; the point estimate and the sign
# test are exact and do not depend on it.
_SEED = 20260917


def _scores(path: Path, metric: str) -> dict[str, float]:
    """query_id -> metric, over queries that carry it."""
    run: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for q in run["per_query"]:
        for container in (q.get("retrieval") or {}, q.get("generation") or {}):
            if metric in container and container[metric] is not None:
                out[q["query_id"]] = float(container[metric])
    return out


def _sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial p at p=0.5 over the discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail: float = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _bootstrap_ci(deltas: list[float]) -> tuple[float, float]:
    rng = random.Random(_SEED)
    n = len(deltas)
    means = []
    for _ in range(_BOOTSTRAP_ROUNDS):
        means.append(sum(rng.choice(deltas) for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * _BOOTSTRAP_ROUNDS)], means[int(0.975 * _BOOTSTRAP_ROUNDS)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="first run JSON")
    parser.add_argument("--b", type=Path, required=True, help="second run JSON")
    parser.add_argument("--metric", default="recall_at_10")
    args = parser.parse_args()

    a, b = _scores(args.a, args.metric), _scores(args.b, args.metric)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit(f"no query_id carries {args.metric} in both runs")

    deltas = [a[q] - b[q] for q in shared]
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = len(deltas) - wins - losses
    mean = sum(deltas) / len(deltas)
    lo, hi = _bootstrap_ci(deltas)

    print(f"metric      {args.metric}")
    print(f"A           {args.a.name}  mean {sum(a[q] for q in shared) / len(shared):.4f}")
    print(f"B           {args.b.name}  mean {sum(b[q] for q in shared) / len(shared):.4f}")
    print(
        f"paired n    {len(shared)}  (A-only {len(set(a) - set(b))}, B-only {len(set(b) - set(a))})"
    )
    print(f"mean delta  {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  (A minus B)")
    print(f"per query   A better {wins}, B better {losses}, tied {ties}")
    print(f"sign test   p = {_sign_test(wins, losses):.4g} over {wins + losses} discordant pairs")


if __name__ == "__main__":
    main()
