"""Generation-side metrics. Ships citation grounding (no LLM needed)."""

from __future__ import annotations

from collections.abc import Iterable


def citation_grounding(
    cited_chunk_ids: Iterable[str], retrieved_chunk_ids: Iterable[str]
) -> float | None:
    """Fraction of citations that reference chunks the retriever actually returned.

    1.0 = all cited chunks are grounded in retrieved context.
    0.0 = all cited chunks are hallucinated (not in retrieved set).
    None = the answer made no citations at all (metric not applicable).
    """
    cited = list(cited_chunk_ids)
    if not cited:
        return None
    retrieved = set(retrieved_chunk_ids)
    grounded = sum(1 for cid in cited if cid in retrieved)
    return grounded / len(cited)
