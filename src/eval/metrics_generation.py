"""Generation-side metrics. Ships citation grounding + refusal detection (no LLM needed)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

# Refusal sentinels:
#   - "Not stated in the provided context.", produced by the answer prompt
#     (src/prompts/library/answer.yaml) when the model refuses.
#   - "I cannot answer this question from the provided corpus.", produced by
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


AnswerOutcome = Literal["refused", "correct", "wrong"]

# Coverage is a fraction of expected facts, so a judge writing 2/3 as "0.66"
# must still count as full coverage at the top of the grid. Mirrors
# judges._GRID_TOLERANCE.
_COVERAGE_TOLERANCE = 0.02


def answer_outcome(
    answer: str | None,
    coverage: float | None,
    *,
    threshold: float = 1.0,
) -> AnswerOutcome:
    """Classify one answer as refused, correct, or wrong.

    `coverage` is an answer_correctness score (recall of expected facts), which
    cannot separate a correct refusal from a confident wrong answer: both score
    0.0. Refusal is therefore decided from the answer text first, and only an
    attempted answer is graded.

    `threshold` is the coverage at or above which an attempt counts as correct.
    The default demands every expected fact, so partial coverage on a multi-fact
    query is wrong; lower it deliberately if partial credit is wanted.
    A missing `coverage` (no judge ran) counts as wrong, never as correct.
    """
    if is_refusal_answer(answer):
        return "refused"
    if coverage is None:
        return "wrong"
    return "correct" if coverage >= threshold - _COVERAGE_TOLERANCE else "wrong"


def outcome_rates(outcomes: Sequence[AnswerOutcome]) -> dict[str, float]:
    """Fraction of each outcome, plus the counts. Empty input gives all zeros.

    `wrong` is the number a prompt or reader change has to hold down: a change
    that lifts mean coverage by converting refusals into wrong answers moves the
    product backwards, which mean coverage alone reports as an improvement.
    """
    total = len(outcomes)
    counts = {
        "refused": sum(1 for o in outcomes if o == "refused"),
        "correct": sum(1 for o in outcomes if o == "correct"),
        "wrong": sum(1 for o in outcomes if o == "wrong"),
    }
    rates = {k: (v / total if total else 0.0) for k, v in counts.items()}
    return {**rates, **{f"n_{k}": float(v) for k, v in counts.items()}, "n": float(total)}
