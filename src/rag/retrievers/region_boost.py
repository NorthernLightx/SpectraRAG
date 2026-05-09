"""Region-number boost: when a query says 'Table 7' or 'Figure 3', promote
chunks whose text starts with that exact label.

ADR 0009 follow-up. The reranker (bge-reranker-v2-m3) doesn't reliably
disambiguate among multiple Table/Figure chunks of the same paper — the
paper has a Table 1 and a Table 8, the query asks about Table 8, and the
cross-encoder still picks Table 1 because the query/document keyword overlap
is similar across both. Run ad4fab3bb28d / q29 demonstrated this.

This wrapper is a deliberate, narrow heuristic: parse the query for explicit
`Table N` / `Figure N` mentions, then reorder the underlying retriever's
results so chunks whose `text` starts with `Table N:` / `Figure N:` come
first. No score mutation — the wrapper only swaps positions, so downstream
score-aware code (rerank score, RRF score) sees the original numbers.

The match is on chunk *text* (the caption), not chunk *id* — the chunk id
uses page-local sequential indexing (`::tab1`, `::tab2`, ...), which doesn't
correspond to the paper's Table N numbering. Caption text is reliable
because `figure_to_chunk` / `table_to_chunk` start the chunk text with
`Figure N: …` / `Table N: …` verbatim from the PDF caption.
"""

from __future__ import annotations

import re

from src.observability.logging import get_logger
from src.rag.retrievers.protocol import Retriever
from src.types import Query, RetrievalResult

_log = get_logger(__name__)

# Captures the (number) group from "Table 7", "Figure 3", "Fig. 12", "Fig 4".
_QUERY_TABLE_RE = re.compile(r"\btable\s+(\d+)\b", re.IGNORECASE)
_QUERY_FIGURE_RE = re.compile(r"\bfigure\s+(\d+)\b|\bfig\.?\s*(\d+)\b", re.IGNORECASE)


def _extract_referenced_numbers(query_text: str) -> tuple[list[int], list[int]]:
    """Return (table_numbers, figure_numbers) referenced in the query.

    Multiple numbers are supported (some queries say "compare Table 3 and
    Table 5"). Returned in source-order; deduplicated.
    """
    table_numbers: list[int] = []
    for match in _QUERY_TABLE_RE.finditer(query_text):
        n = int(match.group(1))
        if n not in table_numbers:
            table_numbers.append(n)

    figure_numbers: list[int] = []
    for match in _QUERY_FIGURE_RE.finditer(query_text):
        # Either group 1 ("Figure N") or group 2 ("Fig. N") matches.
        raw = match.group(1) or match.group(2)
        if raw is None:
            continue
        n = int(raw)
        if n not in figure_numbers:
            figure_numbers.append(n)

    return table_numbers, figure_numbers


def _matches_label(text: str, kind: str, number: int) -> bool:
    """True iff `text` starts with `<kind> <number>:` (case-insensitive).

    Lenient on whitespace and the separator (colon, em-dash, period). Matches
    `figure_to_chunk` / `table_to_chunk` output: PDF caption text starts with
    e.g. "Figure 3: ..." per the regex parsing in figures.py / tables.py.
    """
    head = text.lstrip()
    if not head:
        return False
    pattern = rf"^{kind}\s+{number}\s*[:.—\-]"
    return re.match(pattern, head, re.IGNORECASE) is not None


class RegionNumberBoostRetriever:
    """Wraps a Retriever; reorders results when query references Table N / Figure N.

    Pure post-processor — calls the wrapped retriever, then bubbles matching
    chunks to the top while preserving relative order among matched + among
    non-matched. Drop-in for `Retriever`. ADR 0009 follow-up.
    """

    def __init__(self, *, base: Retriever) -> None:
        self._base = base

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        results = await self._base.retrieve(query)
        table_nums, figure_nums = _extract_referenced_numbers(query.text)
        if not table_nums and not figure_nums:
            return results

        def matches(r: RetrievalResult) -> bool:
            for n in table_nums:
                if _matches_label(r.text, "table", n):
                    return True
            for n in figure_nums:
                # 'figure' covers both "Figure N:" and "Fig. N:" in the chunk
                # text — `figure_to_chunk` writes the verbatim PDF caption so
                # whichever form the paper used is preserved.
                if _matches_label(r.text, "figure", n) or _matches_label(r.text, "fig\\.?", n):
                    return True
            return False

        boosted: list[RetrievalResult] = []
        rest: list[RetrievalResult] = []
        for r in results:
            (boosted if matches(r) else rest).append(r)

        if boosted:
            _log.info(
                "region_boost.applied",
                table_numbers=table_nums,
                figure_numbers=figure_nums,
                boosted_count=len(boosted),
                first_boosted_id=boosted[0].chunk_id,
            )
        return boosted + rest
