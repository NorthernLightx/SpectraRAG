"""Reciprocal Rank Fusion. Score-agnostic combination of ranked lists."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RankedItem:
    """An item with an opaque score from one ranking source."""

    id: str
    score: float


@dataclass(frozen=True)
class FusedItem:
    """An item with its fused RRF score."""

    id: str
    score: float


def reciprocal_rank_fusion(
    lists: list[list[RankedItem]], *, k: int = 60, top_k: int = 50
) -> list[FusedItem]:
    """Fuse ranked lists via RRF. score(d) = sum_i 1 / (k + rank_i(d))."""
    rrf: dict[str, float] = {}
    for ranked in lists:
        for rank, item in enumerate(ranked):
            rrf[item.id] = rrf.get(item.id, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(rrf.items(), key=lambda pair: pair[1], reverse=True)
    return [FusedItem(id=item_id, score=score) for item_id, score in fused[:top_k]]
