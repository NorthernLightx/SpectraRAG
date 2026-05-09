"""Generation-side metrics: citation grounding + latency stats."""

from __future__ import annotations

import pytest

from src.eval.latency import latency_stats
from src.eval.metrics_generation import citation_grounding, is_refusal_answer


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


# is_refusal_answer covers both refusal sentinels (answer-prompt and refusal-gate)
# and is lenient about trailing whitespace / Citations lines / case.


def test_refusal_exact_phrase() -> None:
    assert is_refusal_answer("Not stated in the provided context.")


def test_refusal_with_trailing_citations_line() -> None:
    # Real model output observed in run c92f3f1bee19.
    assert is_refusal_answer("Not stated in the provided context.  \nCitations: None")


def test_refusal_case_insensitive() -> None:
    assert is_refusal_answer("not stated IN THE provided context.")


def test_refusal_alternate_gate_phrase() -> None:
    # The Generator refusal gate (src/rag/generate.py) emits this string.
    assert is_refusal_answer("I cannot answer this question from the provided corpus.")


def test_non_refusal_substantive_answer() -> None:
    assert not is_refusal_answer("The benchmark contains 8 tasks and 65 instances.")


def test_non_refusal_phrase_buried_mid_text_does_not_match() -> None:
    # Strict prefix match — avoids false positives on answers that merely
    # mention the phrase or quote it back.
    assert not is_refusal_answer(
        "The paper says it would be 'Not stated in the provided context.' for missing data."
    )


def test_refusal_empty_or_none() -> None:
    assert not is_refusal_answer(None)
    assert not is_refusal_answer("")
    assert not is_refusal_answer("   \n\n  ")


def test_refusal_with_leading_whitespace() -> None:
    assert is_refusal_answer("\n  Not stated in the provided context.\n")
