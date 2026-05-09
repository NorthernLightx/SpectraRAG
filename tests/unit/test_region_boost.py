"""RegionNumberBoostRetriever — query-grounded promotion of Table N / Figure N chunks."""

from __future__ import annotations

from src.rag.retrievers.region_boost import (
    RegionNumberBoostRetriever,
    _extract_referenced_numbers,
    _matches_label,
)
from src.types import Query, RetrievalResult


class _CannedRetriever:
    """Returns a fixed list of RetrievalResults."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        return list(self._results)


def _result(cid: str, text: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=cid,
        paper_id="p1",
        score=score,
        text=text,
        page_numbers=[1],
        source="pipeline",
    )


# Query-text parsing


def test_extract_table_number() -> None:
    tabs, figs = _extract_referenced_numbers("What's in Table 7 of the paper?")
    assert tabs == [7]
    assert figs == []


def test_extract_figure_number() -> None:
    tabs, figs = _extract_referenced_numbers("What does Figure 3 illustrate?")
    assert tabs == []
    assert figs == [3]


def test_extract_fig_dot_form() -> None:
    tabs, figs = _extract_referenced_numbers("What does Fig. 12 show?")
    assert tabs == []
    assert figs == [12]


def test_extract_multiple_table_numbers_dedup() -> None:
    tabs, figs = _extract_referenced_numbers("compare Table 3 and Table 5 with Table 3 again")
    assert tabs == [3, 5]
    assert figs == []


def test_extract_no_match() -> None:
    tabs, figs = _extract_referenced_numbers("Why is the predictive accuracy goal restricted?")
    assert tabs == []
    assert figs == []


def test_extract_case_insensitive() -> None:
    tabs, figs = _extract_referenced_numbers("WHAT IS IN TABLE 4 AND figure 7?")
    assert tabs == [4]
    assert figs == [7]


# Caption-text label matching (used to identify Table-N / Figure-N chunks)


def test_matches_label_exact() -> None:
    assert _matches_label("Table 7: Properties of surrogate losses.", "table", 7)
    assert _matches_label("Figure 3: Architecture diagram.", "figure", 3)


def test_matches_label_em_dash_separator() -> None:
    assert _matches_label("Figure 3 — Architecture diagram.", "figure", 3)


def test_matches_label_period_separator() -> None:
    assert _matches_label("Figure 3. Architecture diagram.", "figure", 3)


def test_matches_label_leading_whitespace() -> None:
    assert _matches_label("\n   Table 1: A small thing.", "table", 1)


def test_matches_label_wrong_number() -> None:
    """Table 7 query must NOT match Table 1 caption — the bug q29 hit."""
    assert not _matches_label("Table 1: Loss properties.", "table", 7)


def test_matches_label_kind_mismatch() -> None:
    assert not _matches_label("Figure 3: An overview.", "table", 3)


def test_matches_label_buried_text() -> None:
    """The word 'Table 7' mid-sentence is not a label — must not match."""
    assert not _matches_label("As discussed, Table 7 shows the comparison.", "table", 7)


# RegionNumberBoostRetriever — end-to-end


async def test_no_boost_when_query_has_no_region_reference() -> None:
    """Query without 'Table N' / 'Figure N' → results unchanged."""
    base = _CannedRetriever(
        [
            _result("p1::p2::tab1", "Table 1: Properties.", score=0.4),
            _result("p1::p2::c5", "Some text discussing the loss properties.", score=0.9),
        ]
    )
    retriever = RegionNumberBoostRetriever(base=base)
    results = await retriever.retrieve(Query(text="loss properties comparison"))
    assert [r.chunk_id for r in results] == ["p1::p2::tab1", "p1::p2::c5"]


async def test_boost_promotes_matching_table_to_top() -> None:
    """Query says Table 7; chunk whose text starts 'Table 7:' bubbles to position 0."""
    base = _CannedRetriever(
        [
            _result("p1::p2::tab1", "Table 1: Properties.", score=0.9),  # ranked #1, wrong
            _result("p1::p23::c103", "Some text mentioning Table 7.", score=0.5),
            _result("p1::p23::tab1", "Table 7: Janus-Pro ranking.", score=0.4),  # the right one
        ]
    )
    retriever = RegionNumberBoostRetriever(base=base)
    results = await retriever.retrieve(Query(text="What is the rank in Table 7?"))
    assert results[0].chunk_id == "p1::p23::tab1"
    # Non-matching results preserve relative order.
    assert [r.chunk_id for r in results[1:]] == ["p1::p2::tab1", "p1::p23::c103"]


async def test_boost_handles_figure_number() -> None:
    base = _CannedRetriever(
        [
            _result("p1::p1::c0", "Some text.", score=0.9),
            _result("p1::p2::fig1", "Figure 3: An architectural overview.", score=0.4),
        ]
    )
    retriever = RegionNumberBoostRetriever(base=base)
    results = await retriever.retrieve(Query(text="What does Figure 3 illustrate?"))
    assert results[0].chunk_id == "p1::p2::fig1"


async def test_boost_handles_fig_dot_form() -> None:
    base = _CannedRetriever(
        [
            _result("p1::p1::c0", "Body text.", score=0.9),
            _result("p1::p2::fig1", "Fig. 12 — Pipeline overview.", score=0.4),
        ]
    )
    retriever = RegionNumberBoostRetriever(base=base)
    results = await retriever.retrieve(Query(text="What does Fig. 12 show?"))
    assert results[0].chunk_id == "p1::p2::fig1"


async def test_boost_promotes_multiple_matches() -> None:
    """When two chunks match (e.g. Table 3 caption + Table 3 commentary), both rise."""
    base = _CannedRetriever(
        [
            _result("p1::p1::c0", "Body text.", score=0.95),
            _result("p1::p2::tab1", "Table 3: Method comparison.", score=0.5),
            _result("p1::p2::tab2", "Table 3: Continued.", score=0.4),
        ]
    )
    retriever = RegionNumberBoostRetriever(base=base)
    results = await retriever.retrieve(Query(text="what's in Table 3?"))
    # Both Table 3 chunks come first, in their original relative order.
    assert results[0].chunk_id == "p1::p2::tab1"
    assert results[1].chunk_id == "p1::p2::tab2"
    assert results[2].chunk_id == "p1::p1::c0"


async def test_boost_no_op_when_no_match_in_results() -> None:
    """Query mentions Table 99 but no chunk matches — results unchanged."""
    base = _CannedRetriever(
        [
            _result("p1::p2::tab1", "Table 1: Properties.", score=0.9),
            _result("p1::p2::c5", "Body.", score=0.5),
        ]
    )
    retriever = RegionNumberBoostRetriever(base=base)
    results = await retriever.retrieve(Query(text="What's in Table 99?"))
    assert [r.chunk_id for r in results] == ["p1::p2::tab1", "p1::p2::c5"]


async def test_boost_disambiguates_between_tables() -> None:
    """Same paper has Table 1 and Table 8; query for Table 8 must promote tab8 over tab1."""
    base = _CannedRetriever(
        [
            _result("p1::p6::tab1", "Table 1: Loss properties.", score=0.95),
            _result("p1::p23::c103", "On page 23, the ranking is given.", score=0.7),
            _result("p1::p23::tab1", "Table 8: Janus-Pro-7B ranks.", score=0.4),
        ]
    )
    retriever = RegionNumberBoostRetriever(base=base)
    results = await retriever.retrieve(Query(text="rank of Janus-Pro-7B in Table 8"))
    assert results[0].chunk_id == "p1::p23::tab1"
    # Table 1 must NOT also be boosted — wrong number.
    assert results[1].chunk_id == "p1::p6::tab1"
