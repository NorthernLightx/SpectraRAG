"""Routing — classify_query precedence + Query.force_route field per ADR 0008.

Also covers RoutingRetriever dispatch, page-level RRF fusion, and visual-leg
fallback per ADR 0008 §"Architecture" + §"Failure modes".
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.rag.retrievers.routing import (
    Category,
    RoutingRetriever,
    classify_query,
)
from src.types import Query, RetrievalResult


class _RecordingRetriever:
    """Test fake: records call count, returns canned results."""

    def __init__(self, name: str, results: list[RetrievalResult]) -> None:
        self.name = name
        self.results = results
        self.calls = 0

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        self.calls += 1
        return self.results


class _FailingRetriever:
    """Test fake: raises on retrieve. Used for visual-leg failure tests."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        self.calls += 1
        raise self.exc


def _text_chunk(
    chunk_id: str, score: float, page: int = 1, paper: str = "paper1"
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        paper_id=paper,
        score=score,
        text="text chunk",
        page_numbers=[page],
        source="pipeline",
    )


def _visual_page(page: int, score: float, paper: str = "paper1") -> RetrievalResult:
    """Mirrors visual.py's chunk_id convention: <paper>::p<n>::page."""
    return RetrievalResult(
        chunk_id=f"{paper}::p{page}::page",
        paper_id=paper,
        score=score,
        text=f"[Page image {paper} p{page}]",
        page_numbers=[page],
        source="visual",
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        # ---- table (precedence 1) ----
        ("What is the value in Table 4?", "table"),
        ("Show me the cell at row 3 column 2", "table"),
        ("Which row of the dataset has the highest score?", "table"),
        # ---- figure (precedence 2) ----
        ("Show Figure 3 architecture", "figure"),
        ("Look at Fig. 5 in the appendix", "figure"),
        ("What does the chart show?", "figure"),
        ("Describe the diagram on page 7", "figure"),
        # ---- multi_hop (precedence 3) ----
        ("How does method X compare to method Y?", "multi_hop"),
        ("Method A versus Method B", "multi_hop"),
        ("differences between approach 1 and approach 2", "multi_hop"),
        ("Choose between option A or B", "multi_hop"),
        # ---- factual (precedence 4 — numeric span OR ≥2-char acronym) ----
        ("What is the FID score?", "factual"),  # FID = acronym
        ("model achieves 0.85 accuracy", "factual"),  # numeric
        # ---- definitional (precedence 5 — default) ----
        ("What does the model do?", "definitional"),
        ("Explain the methodology", "definitional"),
        ("describe the approach in plain terms", "definitional"),
        # ---- precedence wins ----
        # figure beats multi_hop when both signals present (ADR 0008 §"Classifier")
        ("Compare Figure 3 vs Figure 4", "figure"),
        # table beats multi_hop when both signals present
        ("Compare data in Table 2", "table"),
        # table beats figure if both Table and Figure tokens are present
        ("Figure 3 reproduces Table 4 metrics", "table"),
    ],
)
def test_classify_query(text: str, expected: Category) -> None:
    assert classify_query(text) == expected


def test_classify_query_handles_empty_string() -> None:
    """Edge case: empty input falls through to the default. The Query model
    rejects empty text upstream, but classify_query is a pure function and
    should be safe to call regardless."""
    assert classify_query("") == "definitional"


def test_query_force_route_defaults_to_none() -> None:
    q = Query(text="hello world", top_k=5)
    assert q.force_route is None


def test_query_force_route_accepts_text() -> None:
    q = Query(text="hello", top_k=5, force_route="text")
    assert q.force_route == "text"


def test_query_force_route_accepts_hybrid() -> None:
    q = Query(text="hello", top_k=5, force_route="hybrid")
    assert q.force_route == "hybrid"


def test_query_force_route_rejects_invalid_literal() -> None:
    """Pydantic Literal validation should reject anything outside {text, hybrid, None}."""
    with pytest.raises(ValidationError):
        Query(text="hello", top_k=5, force_route="visual")


# --------------------------------------------------------------------------
# RoutingRetriever
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_routes_text_only_for_definitional_query() -> None:
    """definitional category → text-only path; visual leg never invoked."""
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", 0.9)])
    visual = _RecordingRetriever("visual", [_visual_page(2, 0.5)])
    router = RoutingRetriever(text=text, visual=visual)

    results = await router.retrieve(Query(text="What does the model do?", top_k=5))

    assert text.calls == 1
    assert visual.calls == 0
    assert len(results) == 1
    assert results[0].chunk_id == "paper1::p1::c0"


@pytest.mark.asyncio
async def test_routes_hybrid_for_figure_query() -> None:
    """figure category → hybrid path; both retrievers invoked."""
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", 0.9)])
    visual = _RecordingRetriever("visual", [_visual_page(2, 0.7)])
    router = RoutingRetriever(text=text, visual=visual)

    results = await router.retrieve(Query(text="Show Figure 3 architecture", top_k=5))

    assert text.calls == 1
    assert visual.calls == 1
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_force_route_text_skips_visual_even_for_figure_query() -> None:
    """force_route='text' overrides the classifier — visual never invoked."""
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", 0.9)])
    visual = _RecordingRetriever("visual", [_visual_page(2, 0.7)])
    router = RoutingRetriever(text=text, visual=visual)

    await router.retrieve(Query(text="Show Figure 3", top_k=5, force_route="text"))

    assert text.calls == 1
    assert visual.calls == 0


@pytest.mark.asyncio
async def test_force_route_hybrid_overrides_definitional() -> None:
    """force_route='hybrid' overrides the classifier — visual is invoked."""
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", 0.9)])
    visual = _RecordingRetriever("visual", [_visual_page(2, 0.7)])
    router = RoutingRetriever(text=text, visual=visual)

    await router.retrieve(Query(text="What does the model do?", top_k=5, force_route="hybrid"))

    assert text.calls == 1
    assert visual.calls == 1


@pytest.mark.asyncio
async def test_page_level_fusion_merges_same_page() -> None:
    """text chunk on page 1 + visual page 1 → fused result has ONE entry for page 1.

    Page-keyed deduplication: text chunks normalise to their page-id before RRF
    so the same page from both legs doesn't double-count. The text RetrievalResult
    is preferred when both legs hit a page (chunk-level provenance > page-level)."""
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c5", 0.9, page=1)])
    visual = _RecordingRetriever("visual", [_visual_page(1, 0.8)])
    router = RoutingRetriever(text=text, visual=visual)

    results = await router.retrieve(Query(text="Compare Figure 1 layouts", top_k=5))

    pages = [r.page_numbers[0] for r in results]
    assert pages.count(1) == 1, f"page 1 should appear exactly once, got {pages}"
    page1 = next(r for r in results if r.page_numbers == [1])
    assert page1.source == "pipeline"
    assert page1.chunk_id == "paper1::p1::c5"


@pytest.mark.asyncio
async def test_page_level_fusion_keeps_visual_only_pages() -> None:
    """Visual hit on page 5 (no text hit there) → fused result keeps that page
    via the visual RetrievalResult."""
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", 0.9, page=1)])
    visual = _RecordingRetriever("visual", [_visual_page(5, 0.8)])
    router = RoutingRetriever(text=text, visual=visual)

    results = await router.retrieve(Query(text="Compare Figure in Table 5", top_k=5))

    pages = sorted({r.page_numbers[0] for r in results})
    assert pages == [1, 5]
    page5 = next(r for r in results if r.page_numbers == [5])
    assert page5.source == "visual"
    assert page5.chunk_id == "paper1::p5::page"


@pytest.mark.asyncio
async def test_visual_failure_falls_back_to_text() -> None:
    """Visual leg raises → log warning, return text-only results, do NOT propagate."""
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", 0.9)])
    visual = _FailingRetriever(RuntimeError("CUDA out of memory"))
    router = RoutingRetriever(text=text, visual=visual)

    results = await router.retrieve(Query(text="Show Figure 3", top_k=5))

    assert text.calls == 1
    assert visual.calls == 1  # was attempted
    assert len(results) == 1
    assert results[0].chunk_id == "paper1::p1::c0"
    assert results[0].source == "pipeline"


@pytest.mark.asyncio
async def test_top_k_limits_hybrid_fused_results() -> None:
    """RRF fusion respects top_k. 5 disjoint text + 5 disjoint visual pages, top_k=3 → 3 fused."""
    text = _RecordingRetriever(
        "text",
        [_text_chunk(f"paper1::p{i}::c0", 0.9 - i * 0.01, page=i) for i in range(1, 6)],
    )
    visual = _RecordingRetriever(
        "visual",
        [_visual_page(page=i + 10, score=0.7 - i * 0.01) for i in range(1, 6)],
    )
    router = RoutingRetriever(text=text, visual=visual)

    results = await router.retrieve(Query(text="Compare Figure 1 vs Figure 2", top_k=3))

    assert len(results) == 3
