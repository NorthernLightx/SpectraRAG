"""Talk retrieval scoring: span labels scored against page rankings, beside the
score a random ranking of the talk's segments would get."""

from __future__ import annotations

import random

import pytest

from scripts.score_talk_retrieval import score_query
from src.eval.spans import random_recall_at_k, random_reciprocal_rank
from src.types import MediaSegment
from src.types.eval import GoldenQuery, TimeSpan

SEGMENTS = [MediaSegment(page=n, start_s=10.0 * (n - 1), end_s=10.0 * n) for n in range(1, 11)]


def test_random_recall_is_the_share_of_pages_in_the_top_k() -> None:
    assert random_recall_at_k(n_pages=10, k=1) == pytest.approx(0.1)
    assert random_recall_at_k(n_pages=10, k=3) == pytest.approx(0.3)
    assert random_recall_at_k(n_pages=2, k=3) == 1.0


def test_random_reciprocal_rank_small_cases() -> None:
    assert random_reciprocal_rank(n_pages=1, n_relevant=1) == 1.0
    assert random_reciprocal_rank(n_pages=2, n_relevant=1) == pytest.approx(0.75)
    assert random_reciprocal_rank(n_pages=3, n_relevant=3) == 1.0


def test_random_reciprocal_rank_matches_simulation() -> None:
    rng = random.Random(0)
    n, r, trials = 12, 3, 40_000
    total = 0.0
    for _ in range(trials):
        order = list(range(n))
        rng.shuffle(order)
        total += 1.0 / (min(order.index(i) for i in range(r)) + 1)
    assert random_reciprocal_rank(n_pages=n, n_relevant=r) == pytest.approx(
        total / trials, abs=0.01
    )


def test_score_query_projects_chunks_to_pages_and_scores_them() -> None:
    q = GoldenQuery(
        query_id="q",
        text="?",
        paper_id="talk",
        category="figure",
        relevant_spans=[TimeSpan(start_s=31.0, end_s=39.0)],  # page 4
    )
    ranked = ["talk::p2::c3", "talk::p2::c4", "talk::p4::c7", "talk::p9::c12"]
    s = score_query(q, SEGMENTS, ranked)
    assert s["relevant_pages"] == [4]
    assert (s["recall_at_1"], s["recall_at_3"], s["mrr"]) == (0.0, 1.0, 0.5)
    assert s["random_recall_at_1"] == pytest.approx(0.1)
    # Three distinct pages ranked, so random gets three ranks.
    assert s["random_mrr"] == pytest.approx(
        random_reciprocal_rank(n_pages=10, n_relevant=1, depth=3)
    )


def test_random_reciprocal_rank_counts_only_ranks_within_the_depth() -> None:
    # A top-1 list finds the one relevant page of ten with probability 1/10.
    assert random_reciprocal_rank(n_pages=10, n_relevant=1, depth=1) == pytest.approx(0.1)
    assert random_reciprocal_rank(n_pages=3, n_relevant=1, depth=5) == pytest.approx(
        random_reciprocal_rank(n_pages=3, n_relevant=1)
    )


def test_score_query_compares_with_random_at_the_same_depth() -> None:
    # A top-2 list over a ten-page talk: random gets two ranks, not ten.
    q = GoldenQuery(
        query_id="q",
        text="?",
        paper_id="talk",
        category="factual",
        relevant_spans=[TimeSpan(start_s=31.0, end_s=39.0)],
    )
    s = score_query(q, SEGMENTS, ["talk::p1::c0", "talk::p2::c1"])
    assert s["depth"] == 2
    assert s["random_recall_at_3"] == pytest.approx(0.2)
    assert s["random_mrr"] == pytest.approx(
        random_reciprocal_rank(n_pages=10, n_relevant=1, depth=2)
    )
