"""ADR 0024 route-by-fit page selector in the MMLongBench-Doc QA generation
harness (scripts/experiments/run_mmlb_qa.py:route_pages_by_fit).

Pure list/int math: no Ollama, no Qdrant, no GPU, no disk (the helper takes the
page count as a plain int and the full page list as data). Asserts the closed-
interval fit test: page_count <= budget feeds the whole document, else the
top-k RAG fallback (select_pages) fires verbatim.
"""

from __future__ import annotations

from scripts.experiments.run_mmlb_qa import route_pages_by_fit, select_pages
from scripts.rescore_mmlb_pages import Page

# A whole-doc page list (what the caller resolves from data/pages/<paper>) and a
# fused ranking whose top-k cut is deliberately a DIFFERENT, smaller set, so a
# test can tell the two arms apart by their output alone.
_PAPER = "doc"
_ALL_PAGES: list[Page] = [(_PAPER, n) for n in range(1, 11)]  # 10-page doc
_FUSED: list[str] = [
    "doc::p3::c0",
    "doc::p3::c1",  # dup page 3 -> dropped by rank-order dedup
    "doc::p7::c0",
    "doc::p1::c0",
    "doc::p9::c0",
]
_TOP_K = 2


def test_fits_budget_returns_all_pages() -> None:
    """page_count < budget: feed the whole document, not the top-k cut."""
    out = route_pages_by_fit(10, 50, _ALL_PAGES, _FUSED, _TOP_K)
    assert out == _ALL_PAGES
    assert out is _ALL_PAGES  # whole-doc arm returns the list as-is, no copy/truncation


def test_exceeds_budget_returns_select_pages() -> None:
    """page_count > budget: fall back to EXACTLY select_pages(fused, top_k)."""
    out = route_pages_by_fit(100, 50, _ALL_PAGES, _FUSED, _TOP_K)
    assert out == select_pages(_FUSED, _TOP_K)
    assert out == [("doc", 3), ("doc", 7)]  # rank-order dedup, truncated to top_k


def test_boundary_equal_routes_whole_doc() -> None:
    """page_count == budget is the CLOSED-interval boundary: whole-doc, not RAG."""
    out = route_pages_by_fit(50, 50, _ALL_PAGES, _FUSED, _TOP_K)
    assert out == _ALL_PAGES
    assert out != select_pages(_FUSED, _TOP_K)


def test_degenerate_budget_zero() -> None:
    """budget=0: no real doc fits (page_count>=1), so always RAG; and the
    page_count==0 corner still routes whole-doc by the closed interval."""
    assert route_pages_by_fit(1, 0, _ALL_PAGES, _FUSED, _TOP_K) == select_pages(_FUSED, _TOP_K)
    assert route_pages_by_fit(0, 0, [], _FUSED, _TOP_K) == []  # 0 <= 0 -> whole-doc (empty list)


def test_degenerate_budget_one() -> None:
    """budget=1: only a single-page doc fits; a 2-page doc routes to RAG."""
    one_page: list[Page] = [(_PAPER, 1)]
    assert route_pages_by_fit(1, 1, one_page, _FUSED, _TOP_K) == one_page  # 1 <= 1 whole-doc
    assert route_pages_by_fit(2, 1, _ALL_PAGES, _FUSED, _TOP_K) == select_pages(_FUSED, _TOP_K)


def test_output_changes_when_page_count_crosses_budget() -> None:
    """Dumbest sanity check: hold everything fixed, move page_count across the
    budget, and the output must flip between the two arms (a constant output
    would mean the routing condition is dead)."""
    budget = 20
    below = route_pages_by_fit(budget, budget, _ALL_PAGES, _FUSED, _TOP_K)  # at boundary: whole-doc
    above = route_pages_by_fit(budget + 1, budget, _ALL_PAGES, _FUSED, _TOP_K)  # over: RAG
    assert below == _ALL_PAGES
    assert above == select_pages(_FUSED, _TOP_K)
    assert below != above  # the input change produced an output change
