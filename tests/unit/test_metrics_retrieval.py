"""Retrieval metrics: nDCG@k, recall@k, MRR. Pure-function unit tests."""

from __future__ import annotations

import math

import pytest

from src.eval.metrics_retrieval import ndcg_at_k, recall_at_k, reciprocal_rank


def test_ndcg_perfect_when_all_relevant_at_top() -> None:
    relevant = {"c1", "c2"}
    ranked = ["c1", "c2", "c3", "c4", "c5"]
    assert ndcg_at_k(relevant, ranked, k=5) == pytest.approx(1.0)


def test_ndcg_decreases_when_relevant_pushed_down() -> None:
    relevant = {"c5"}
    top_ranked = ["c5", "x", "x", "x", "x"]
    bot_ranked = ["x", "x", "x", "x", "c5"]
    assert ndcg_at_k(relevant, top_ranked, k=5) > ndcg_at_k(relevant, bot_ranked, k=5)


def test_ndcg_specific_value() -> None:
    """Single relevant at rank 2 → DCG = 1/log2(3); IDCG = 1/log2(2) = 1."""
    relevant = {"c2"}
    ranked = ["c1", "c2", "c3"]
    expected = (1.0 / math.log2(3)) / 1.0
    assert ndcg_at_k(relevant, ranked, k=3) == pytest.approx(expected)


def test_ndcg_zero_when_no_relevant_in_top_k() -> None:
    assert ndcg_at_k({"c99"}, ["c1", "c2", "c3"], k=3) == 0.0


def test_ndcg_zero_when_relevant_set_empty() -> None:
    assert ndcg_at_k(set(), ["c1", "c2"], k=5) == 0.0


def test_recall_at_k_proportion_of_relevant_retrieved() -> None:
    relevant = {"c1", "c2", "c3"}
    ranked = ["c1", "c2", "x", "y", "z"]
    assert recall_at_k(relevant, ranked, k=5) == pytest.approx(2 / 3)


def test_recall_at_k_caps_at_k() -> None:
    relevant = {"c1", "c2", "c3"}
    ranked = ["x", "x", "x", "c1", "c2", "c3"]
    assert recall_at_k(relevant, ranked, k=3) == 0.0
    assert recall_at_k(relevant, ranked, k=6) == 1.0


def test_recall_zero_when_relevant_set_empty() -> None:
    assert recall_at_k(set(), ["c1"], k=10) == 0.0


def test_reciprocal_rank_first_position() -> None:
    assert reciprocal_rank({"c1"}, ["c1", "c2", "c3"]) == pytest.approx(1.0)


def test_reciprocal_rank_fifth_position() -> None:
    assert reciprocal_rank({"cX"}, ["c1", "c2", "c3", "c4", "cX"]) == pytest.approx(1 / 5)


def test_reciprocal_rank_zero_when_not_found() -> None:
    assert reciprocal_rank({"cX"}, ["c1", "c2"]) == 0.0


def test_reciprocal_rank_zero_when_relevant_empty() -> None:
    assert reciprocal_rank(set(), ["c1"]) == 0.0
