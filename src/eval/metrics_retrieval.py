"""Retrieval metrics: pure functions over (relevant_ids, ranked_ids).

Binary relevance only — extend to graded later if golden labels grow nuance.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def _to_set(ids: Iterable[str]) -> set[str]:
    return set(ids)


def ndcg_at_k(relevant_ids: Iterable[str], ranked_ids: list[str], k: int = 5) -> float:
    """Binary-relevance nDCG@k.

    Returns 0.0 when there are no relevant ids (matches convention; out-of-corpus
    queries should be evaluated separately).
    """
    relevant = _to_set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item_id in enumerate(ranked_ids[:k])
        if item_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(relevant_ids: Iterable[str], ranked_ids: list[str], k: int = 10) -> float:
    """|relevant ∩ top_k| / |relevant|. 0.0 when relevant is empty."""
    relevant = _to_set(relevant_ids)
    if not relevant:
        return 0.0
    retrieved = set(ranked_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def reciprocal_rank(relevant_ids: Iterable[str], ranked_ids: list[str]) -> float:
    """1 / rank of first relevant item (1-indexed). 0.0 if no relevant retrieved."""
    relevant = _to_set(relevant_ids)
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0
