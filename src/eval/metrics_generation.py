"""Generation-side metrics. Ships citation grounding + refusal detection (no LLM needed)."""

from __future__ import annotations

from collections.abc import Iterable

# Refusal sentinels:
#   - "Not stated in the provided context." — produced by the answer prompt
#     (src/prompts/library/answer.yaml) when the model refuses.
#   - "I cannot answer this question from the provided corpus." — produced by
#     Generator's refusal gate (src/rag/generate.py) when rerank scores are
#     below the configured threshold.
# Both must be detected so the eval scoring is consistent across paths.
_REFUSAL_PREFIXES = (
    "not stated in the provided context",
    "i cannot answer this question from the provided corpus",
)


def is_refusal_answer(answer: str | None) -> bool:
    """True iff `answer` begins with one of the documented refusal sentinels.

    Lenient about trailing whitespace/punctuation/citations: real model output
    looks like `"Not stated in the provided context.  \\nCitations: None"`,
    which still qualifies. Strict prefix match avoids false positives on
    answers that merely mention the phrase mid-sentence.
    """
    if not answer:
        return False
    head = answer.strip().lower()
    return any(head.startswith(prefix) for prefix in _REFUSAL_PREFIXES)


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
