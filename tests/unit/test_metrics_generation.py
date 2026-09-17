"""Generation-side metrics: citation grounding + latency stats."""

from __future__ import annotations

import pytest

from src.eval.latency import latency_stats
from src.eval.metrics_generation import (
    answer_outcome,
    citation_grounding,
    is_refusal_answer,
    outcome_rates,
)


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


# answer_outcome separates the two cases coverage alone collapses to 0.0:
# a correct refusal and a confident wrong answer.


def test_refusal_and_wrong_answer_both_score_zero_coverage() -> None:
    """The reason the outcome metric exists. Same coverage, opposite outcomes."""
    refusal = answer_outcome("Not stated in the provided context.", 0.0)
    confident_miss = answer_outcome("The value is 42%.", 0.0)

    assert refusal == "refused"
    assert confident_miss == "wrong"


def test_refusal_wins_over_coverage() -> None:
    """A judge that scored a refusal above zero does not make it an attempt."""
    assert answer_outcome("Not stated in the provided context.", 1.0) == "refused"


@pytest.mark.parametrize(
    ("coverage", "expected"),
    [(1.0, "correct"), (0.99, "correct"), (0.67, "wrong"), (0.0, "wrong"), (None, "wrong")],
)
def test_attempt_graded_against_full_coverage(coverage: float | None, expected: str) -> None:
    assert answer_outcome("Some answer.", coverage) == expected


def test_threshold_allows_partial_credit_when_asked_for() -> None:
    assert answer_outcome("Some answer.", 0.5, threshold=0.5) == "correct"


def test_outcome_rates_reports_fractions_and_counts() -> None:
    rates = outcome_rates(["refused", "refused", "correct", "wrong"])

    assert rates["refused"] == pytest.approx(0.5)
    assert rates["correct"] == pytest.approx(0.25)
    assert rates["wrong"] == pytest.approx(0.25)
    assert rates["n_refused"] == pytest.approx(2.0)
    assert rates["n"] == pytest.approx(4.0)


def test_outcome_rates_on_empty_input() -> None:
    assert outcome_rates([]) == {
        "refused": 0.0,
        "correct": 0.0,
        "wrong": 0.0,
        "n_refused": 0.0,
        "n_correct": 0.0,
        "n_wrong": 0.0,
        "n": 0.0,
    }


def test_trading_refusals_for_attempts_shows_up_as_wrong() -> None:
    """A change that lifts mean coverage by answering more can raise the wrong
    count at the same time. Mean coverage hides that; these rates do not."""
    before = outcome_rates(["refused", "refused", "refused", "correct"])
    after = outcome_rates(["wrong", "wrong", "correct", "correct"])

    mean_before = (0.0 + 0.0 + 0.0 + 1.0) / 4
    mean_after = (0.0 + 0.0 + 1.0 + 1.0) / 4
    assert mean_after > mean_before  # coverage calls it an improvement
    assert after["wrong"] > before["wrong"]  # the outcome metric does not
