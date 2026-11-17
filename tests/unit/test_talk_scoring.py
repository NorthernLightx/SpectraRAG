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
    assert s is not None
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
    assert s is not None
    assert s["depth"] == 2
    assert s["random_recall_at_3"] == pytest.approx(0.2)
    assert s["random_mrr"] == pytest.approx(
        random_reciprocal_rank(n_pages=10, n_relevant=1, depth=2)
    )


def test_score_query_buckets_evidence_by_how_many_pages_it_covers() -> None:
    # Wide evidence is easy even for a random ranking, so results are split by it.
    def width(start: float, end: float) -> str:
        q = GoldenQuery(
            query_id="q",
            text="?",
            paper_id="talk",
            category="factual",
            relevant_spans=[TimeSpan(start_s=start, end_s=end)],
        )
        s = score_query(q, SEGMENTS, ["talk::p1::c0"])
        assert s is not None
        return str(s["width"])

    assert width(31.0, 39.0) == "1 page"
    assert width(12.0, 38.0) == "2-3 pages"
    assert width(0.0, 100.0) == "4+ pages"


def test_random_hit_is_the_chance_that_any_relevant_page_makes_the_top_k() -> None:
    from src.eval.spans import random_hit_at_k

    assert random_hit_at_k(n_pages=10, n_relevant=1, k=1) == pytest.approx(0.1)
    assert random_hit_at_k(n_pages=10, n_relevant=2, k=3) == pytest.approx(1 - 56 / 120)
    assert random_hit_at_k(n_pages=3, n_relevant=1, k=5) == 1.0


def test_score_query_reports_hits_for_wide_evidence() -> None:
    # Evidence over pages 4-6: recall@1 can reach only 1/3, a hit counts in full.
    q = GoldenQuery(
        query_id="q",
        text="?",
        paper_id="talk",
        category="factual",
        relevant_spans=[TimeSpan(start_s=31.0, end_s=59.0)],
    )
    s = score_query(q, SEGMENTS, ["talk::p5::c0", "talk::p1::c1", "talk::p9::c2"])
    assert s is not None
    assert s["relevant_pages"] == [4, 5, 6]
    assert s["recall_at_1"] == pytest.approx(1 / 3)
    assert (s["hit_at_1"], s["hit_at_3"]) == (1.0, 1.0)


def _q(start: float, end: float) -> GoldenQuery:
    return GoldenQuery(
        query_id="q",
        text="?",
        paper_id="talk",
        category="factual",
        relevant_spans=[TimeSpan(start_s=start, end_s=end)],
    )


def test_a_query_with_no_page_is_not_scored_as_a_miss() -> None:
    assert score_query(_q(200.0, 210.0), SEGMENTS, ["talk::p1::c0"]) is None


def test_a_ranking_from_another_recording_is_refused() -> None:
    with pytest.raises(ValueError, match="other recordings"):
        score_query(_q(31.0, 39.0), SEGMENTS, ["talk::p4::c0", "other::p2::c9"])


def test_a_ranked_page_missing_from_the_manifest_is_refused() -> None:
    # The run and the manifests come from different segmentations.
    with pytest.raises(ValueError, match="manifest"):
        score_query(_q(31.0, 39.0), SEGMENTS, ["talk::p4::c0", "talk::p12::c30"])


def test_the_in_order_baseline_ranks_pages_by_time() -> None:
    # Always answering with the first pages scores well on title-slide questions.
    ranked = [f"talk::p{p}::c{p}" for p in (9, 8, 7, 6, 5)]
    s = score_query(_q(31.0, 39.0), SEGMENTS, ranked)  # page 4
    assert s is not None
    assert s["in_order_mrr"] == pytest.approx(0.25)
    assert s["in_order_hit_at_3"] == 0.0
    s = score_query(_q(1.0, 9.0), SEGMENTS, ranked)  # page 1
    assert s is not None and s["in_order_hit_at_1"] == 1.0
