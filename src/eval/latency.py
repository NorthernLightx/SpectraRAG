"""Latency-distribution helpers for eval reporting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyStats:
    """Summary statistics over a list of latency samples (milliseconds)."""

    n: int
    p50_ms: float
    p95_ms: float
    mean_ms: float


def latency_stats(latencies_ms: Sequence[int | float]) -> LatencyStats:
    """Nearest-rank percentile for small samples; mean over the same sequence."""
    if not latencies_ms:
        return LatencyStats(n=0, p50_ms=0.0, p95_ms=0.0, mean_ms=0.0)
    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)
    p50_index = max(0, min(n - 1, round(0.50 * (n - 1))))
    p95_index = max(0, min(n - 1, round(0.95 * (n - 1))))
    return LatencyStats(
        n=n,
        p50_ms=float(sorted_lats[p50_index]),
        p95_ms=float(sorted_lats[p95_index]),
        mean_ms=sum(sorted_lats) / n,
    )
