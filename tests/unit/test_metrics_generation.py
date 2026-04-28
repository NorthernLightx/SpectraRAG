"""Generation-side metrics: citation grounding + latency stats."""

from __future__ import annotations

import pytest

from src.eval.latency import latency_stats
from src.eval.metrics_generation import citation_grounding


def test_citation_grounding_all_grounded() -> None:
    assert citation_grounding(["c1", "c2"], ["c1", "c2", "c3"]) == pytest.approx(1.0)


def test_citation_grounding_partial() -> None:
    # 1 of 3 cited chunks was actually retrieved
    assert citation_grounding(["c1", "x", "y"], ["c1", "c2"]) == pytest.approx(1 / 3)


def test_citation_grounding_none_when_no_citations() -> None:
    assert citation_grounding([], ["c1", "c2"]) is None


def test_citation_grounding_zero_when_all_hallucinated() -> None:
    assert citation_grounding(["x", "y"], ["c1", "c2"]) == 0.0


def test_latency_stats_with_distribution() -> None:
    stats = latency_stats([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    assert stats.n == 10
    assert stats.p50_ms == 500.0  # nearest-rank index 5 (rounded)
    assert stats.p95_ms == 1000.0
    assert stats.mean_ms == pytest.approx(550.0)


def test_latency_stats_empty() -> None:
    stats = latency_stats([])
    assert stats.n == 0
    assert stats.p50_ms == stats.p95_ms == stats.mean_ms == 0.0


def test_latency_stats_single_sample() -> None:
    stats = latency_stats([123])
    assert stats.n == 1
    assert stats.p50_ms == 123.0
    assert stats.p95_ms == 123.0
    assert stats.mean_ms == 123.0
