"""Reciprocal Rank Fusion: combines ranked lists into a single score."""

from __future__ import annotations

from src.rag.hybrid import RankedItem, reciprocal_rank_fusion


def test_rrf_promotes_items_present_in_multiple_lists() -> None:
    dense = [RankedItem(id="c1", score=0.9), RankedItem(id="c2", score=0.8)]
    sparse = [RankedItem(id="c2", score=5.0), RankedItem(id="c3", score=3.0)]

    fused = reciprocal_rank_fusion([dense, sparse], k=60, top_k=3)

    fused_ids = [item.id for item in fused]
    assert fused_ids[0] == "c2"
    assert set(fused_ids) == {"c1", "c2", "c3"}


def test_rrf_with_single_list_preserves_order() -> None:
    items = [RankedItem(id="a", score=10), RankedItem(id="b", score=5)]
    fused = reciprocal_rank_fusion([items], k=60, top_k=10)
    assert [it.id for it in fused] == ["a", "b"]


def test_rrf_handles_empty_lists() -> None:
    assert reciprocal_rank_fusion([], k=60, top_k=10) == []
    assert reciprocal_rank_fusion([[]], k=60, top_k=10) == []


def test_rrf_caps_to_top_k() -> None:
    items = [RankedItem(id=f"c{i}", score=float(i)) for i in range(20)]
    assert len(reciprocal_rank_fusion([items], k=60, top_k=5)) == 5
