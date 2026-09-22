"""Routing — classify_query precedence + Query.force_route field per ADR 0008.

Also covers RoutingRetriever dispatch, page-level RRF fusion, and visual-leg
fallback per ADR 0008 §"Architecture" + §"Failure modes".
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.rag.hybrid import RankedItem, reciprocal_rank_fusion
from src.rag.retrievers.routing import (
    Category,
    RoutingRetriever,
    classify_query,
    get_last_routing_info,
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
        score_kind="rerank",
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
        score_kind="maxsim",
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
        ("show me the graph", "figure"),
        ("summarize graphs in the paper", "figure"),
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


def test_query_force_route_accepts_visual() -> None:
    q = Query(text="hello", top_k=5, force_route="visual")
    assert q.force_route == "visual"


def test_query_force_route_rejects_invalid_literal() -> None:
    """Pydantic Literal validation should reject anything outside {text, visual, hybrid, None}."""
    with pytest.raises(ValidationError):
        Query(text="hello", top_k=5, force_route="audio")


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
async def test_force_route_visual_runs_visual_only() -> None:
    """force_route='visual' runs ONLY the visual leg — no text leg, no fusion."""
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", 0.9)])
    visual = _RecordingRetriever("visual", [_visual_page(2, 0.7)])
    router = RoutingRetriever(text=text, visual=visual)

    results = await router.retrieve(Query(text="anything", top_k=5, force_route="visual"))

    assert visual.calls == 1
    assert text.calls == 0  # text leg never invoked
    assert [r.chunk_id for r in results] == ["paper1::p2::page"]
    info = get_last_routing_info()
    assert info is not None
    assert info.path == "visual"
    assert info.forced is True


@pytest.mark.asyncio
async def test_force_route_visual_degrades_to_text_on_visual_failure() -> None:
    """A visual-leg failure on the visual-only route degrades to text-only."""
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", 0.9)])
    visual = _FailingRetriever(RuntimeError("GPU OOM"))
    router = RoutingRetriever(text=text, visual=visual)

    results = await router.retrieve(Query(text="anything", top_k=5, force_route="visual"))

    assert visual.calls == 1
    assert text.calls == 1  # degraded to the text leg
    assert [r.chunk_id for r in results] == ["paper1::p1::c0"]
    info = get_last_routing_info()
    assert info is not None
    assert info.path == "text"
    assert info.visual_failed is True


class _FailingClassifier:
    """LLM classifier whose backend is unreachable — classify() raises. Models
    the Cloud Run case where the keyless wiring built an Ollama-backed
    classifier but no Ollama is reachable (ADR 0013 / ADR 0028)."""

    def __init__(self) -> None:
        self.calls = 0

    async def classify(self, query: str) -> Category:
        self.calls += 1
        raise RuntimeError("classifier backend unreachable")


@pytest.mark.asyncio
async def test_classifier_failure_falls_back_to_regex_text() -> None:
    """A failing classifier must not take down retrieval: fall back to the regex
    classifier. A definitional query then routes text-only (ADR 0028)."""
    classifier = _FailingClassifier()
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", 0.9)])
    visual = _RecordingRetriever("visual", [_visual_page(2, 0.5)])
    router = RoutingRetriever(text=text, visual=visual, classifier=classifier)  # type: ignore[arg-type]

    results = await router.retrieve(Query(text="What is exploration hacking?", top_k=5))

    assert classifier.calls == 1  # attempted, then fell back
    assert text.calls == 1
    assert visual.calls == 0  # regex → definitional → text-only
    assert [r.chunk_id for r in results] == ["paper1::p1::c0"]
    info = get_last_routing_info()
    assert info is not None
    assert info.category == "definitional"
    assert info.path == "text"


@pytest.mark.asyncio
async def test_classifier_failure_regex_still_routes_figure_to_hybrid() -> None:
    """Post-fallback the regex still routes correctly: a 'Figure N' query goes
    hybrid, so both legs fire."""
    classifier = _FailingClassifier()
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", 0.9)])
    visual = _RecordingRetriever("visual", [_visual_page(2, 0.7)])
    router = RoutingRetriever(text=text, visual=visual, classifier=classifier)  # type: ignore[arg-type]

    await router.retrieve(Query(text="What does Figure 3 show?", top_k=5))

    assert classifier.calls == 1
    assert text.calls == 1
    assert visual.calls == 1
    info = get_last_routing_info()
    assert info is not None
    assert info.category == "figure"
    assert info.path == "hybrid"


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


# --------------------------------------------------------------------------
# ADR 0023: visual-favoring weighted page-level RRF
# --------------------------------------------------------------------------


def test_weighted_fusion_default_weight_equals_equal_weight_rrf() -> None:
    """w_visual=1.0 must reproduce the shipped equal-weight RRF page order exactly.

    Ground truth is src.rag.hybrid.reciprocal_rank_fusion over the same per-leg
    page lists (the dedup + RankedItem path _fuse_page_level used before ADR
    0023). Overlapping + disjoint pages, ties, and visual-only pages are all
    present so the comparison exercises the insertion-order tie break too.
    """
    # text leg: pages 1,2,3 ; visual leg: 3,4,1 (overlap on 1 and 3, ranks differ)
    text_results = [
        _text_chunk("paper1::p1::c0", 0.9, page=1),
        _text_chunk("paper1::p2::c0", 0.8, page=2),
        _text_chunk("paper1::p3::c0", 0.7, page=3),
    ]
    visual_results = [
        _visual_page(3, 0.9),
        _visual_page(4, 0.8),
        _visual_page(1, 0.7),
    ]
    router = RoutingRetriever(
        text=_RecordingRetriever("t", []), visual=_RecordingRetriever("v", [])
    )

    fused = router._fuse_page_level(text_results, visual_results, top_k=10)
    got_pages = [r.page_numbers[0] for r in fused]

    # Oracle: plain RRF over the same first-appearance page lists.
    text_pages = [RankedItem(id="paper1::p1", score=1.0)] + [
        RankedItem(id=f"paper1::p{n}", score=1.0) for n in (2, 3)
    ]
    visual_pages = [RankedItem(id=f"paper1::p{n}", score=1.0) for n in (3, 4, 1)]
    oracle = reciprocal_rank_fusion([text_pages, visual_pages], k=60, top_k=10)
    oracle_pages = [int(item.id.split("::p")[1]) for item in oracle]

    assert got_pages == oracle_pages


def test_weighted_fusion_high_visual_weight_reorders_toward_visual() -> None:
    """A large visual weight promotes a visual-only page above a text-only page.

    Legs: text=[p1,p2] (p2 also visual-ranked #2), visual=[p3,p2]. p3 is
    visual-only, p1 text-only. At w=1.0 they tie on the RRF score 1/61 and
    insertion order keeps text's p1 first (-> p2,p1,p3). At w=10 the visual leg
    dominates: p3 (10/61) overtakes p1 (1/61) -> p2,p3,p1. The reordering is
    the load-bearing assertion: same inputs, weight alone flips p1 vs p3.
    """
    text_results = [
        _text_chunk("paper1::p1::c0", 0.9, page=1),
        _text_chunk("paper1::p2::c0", 0.8, page=2),
    ]
    visual_results = [
        _visual_page(3, 0.9),
        _visual_page(2, 0.8),
    ]
    text_ret = _RecordingRetriever("t", [])
    visual_ret = _RecordingRetriever("v", [])

    equal = RoutingRetriever(text=text_ret, visual=visual_ret, visual_fusion_weight=1.0)
    weighted = RoutingRetriever(text=text_ret, visual=visual_ret, visual_fusion_weight=10.0)

    equal_pages = [
        r.page_numbers[0] for r in equal._fuse_page_level(text_results, visual_results, top_k=10)
    ]
    weighted_pages = [
        r.page_numbers[0] for r in weighted._fuse_page_level(text_results, visual_results, top_k=10)
    ]

    assert equal_pages == [2, 1, 3]
    assert weighted_pages == [2, 3, 1]


def test_weighted_fusion_default_constructor_weight_is_one() -> None:
    """No visual_fusion_weight arg => 1.0 (the conservative ADR 0023 default)."""
    router = RoutingRetriever(
        text=_RecordingRetriever("t", []), visual=_RecordingRetriever("v", [])
    )
    assert router._visual_fusion_weight == 1.0


# ADR 0010: cascade mode — text first, fall back to visual only on uncertainty.


def test_cascade_mode_requires_threshold() -> None:
    """cascade mode without a threshold is a config error — fail fast at init."""
    text = _RecordingRetriever("text", [])
    visual = _RecordingRetriever("visual", [])
    with pytest.raises(ValueError, match="cascade_confidence_threshold"):
        RoutingRetriever(text=text, visual=visual, mode="cascade")


@pytest.mark.asyncio
async def test_cascade_confident_text_skips_visual_call() -> None:
    """Top-1 text score ≥ threshold → return text-only, visual leg NOT called."""
    text = _RecordingRetriever(
        "text",
        [_text_chunk("paper1::p1::c0", score=0.9, page=1)],
    )
    visual = _RecordingRetriever("visual", [_visual_page(page=1, score=0.5)])
    router = RoutingRetriever(
        text=text, visual=visual, mode="cascade", cascade_confidence_threshold=0.5
    )

    results = await router.retrieve(Query(text="What is X?"))

    assert text.calls == 1
    assert visual.calls == 0  # critical: visual leg skipped on confident text
    assert results[0].chunk_id == "paper1::p1::c0"


@pytest.mark.asyncio
async def test_cascade_uncertain_text_invokes_visual_and_fuses() -> None:
    """Top-1 text score < threshold → run visual leg, RRF-fuse."""
    text = _RecordingRetriever(
        "text",
        [_text_chunk("paper1::p1::c0", score=0.3, page=1)],
    )
    visual = _RecordingRetriever("visual", [_visual_page(page=2, score=0.8)])
    router = RoutingRetriever(
        text=text, visual=visual, mode="cascade", cascade_confidence_threshold=0.5
    )

    results = await router.retrieve(Query(text="What is X?"))

    assert text.calls == 1
    assert visual.calls == 1  # visual leg ran
    # RRF over two disjoint legs returns both; with top_k default=5, both fit.
    chunk_ids = {r.chunk_id for r in results}
    assert "paper1::p1::c0" in chunk_ids
    assert "paper1::p2::page" in chunk_ids


@pytest.mark.asyncio
async def test_cascade_force_route_text_skips_visual() -> None:
    """Explicit force_route='text' overrides confidence — never call visual."""
    text = _RecordingRetriever(
        "text",
        [_text_chunk("paper1::p1::c0", score=0.1, page=1)],  # low score, would normally fall back
    )
    visual = _RecordingRetriever("visual", [_visual_page(page=1, score=0.9)])
    router = RoutingRetriever(
        text=text, visual=visual, mode="cascade", cascade_confidence_threshold=0.5
    )

    results = await router.retrieve(Query(text="X?", force_route="text"))

    assert visual.calls == 0
    assert results[0].chunk_id == "paper1::p1::c0"


@pytest.mark.asyncio
async def test_cascade_force_route_hybrid_invokes_visual() -> None:
    """force_route='hybrid' overrides confidence — always call visual."""
    text = _RecordingRetriever(
        "text",
        [
            _text_chunk("paper1::p1::c0", score=0.99, page=1)
        ],  # high score, would normally skip visual
    )
    visual = _RecordingRetriever("visual", [_visual_page(page=2, score=0.5)])
    router = RoutingRetriever(
        text=text, visual=visual, mode="cascade", cascade_confidence_threshold=0.5
    )

    await router.retrieve(Query(text="X?", force_route="hybrid"))

    assert visual.calls == 1


@pytest.mark.asyncio
async def test_cascade_visual_failure_falls_back_to_text() -> None:
    """If the visual leg crashes, cascade returns text-only — same graceful
    degrade as the category-mode hybrid path."""
    text = _RecordingRetriever(
        "text",
        [_text_chunk("paper1::p1::c0", score=0.3, page=1)],
    )
    failing_visual = _FailingRetriever(RuntimeError("CUDA OOM"))
    router = RoutingRetriever(
        text=text, visual=failing_visual, mode="cascade", cascade_confidence_threshold=0.5
    )

    results = await router.retrieve(Query(text="X?"))

    assert failing_visual.calls == 1  # we tried
    assert results[0].chunk_id == "paper1::p1::c0"  # but degraded to text


@pytest.mark.asyncio
async def test_cascade_empty_text_results_falls_back_to_hybrid() -> None:
    """No text results → top_score=0.0 < threshold → invoke visual fallback."""
    text = _RecordingRetriever("text", [])
    visual = _RecordingRetriever("visual", [_visual_page(page=1, score=0.7)])
    router = RoutingRetriever(
        text=text, visual=visual, mode="cascade", cascade_confidence_threshold=0.5
    )

    results = await router.retrieve(Query(text="X?"))

    assert visual.calls == 1
    assert results[0].chunk_id == "paper1::p1::page"


@pytest.mark.asyncio
async def test_category_mode_default_unchanged() -> None:
    """Sanity: default mode='category' preserves the exact prior dispatch behavior."""
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", score=0.3, page=1)])
    visual = _RecordingRetriever("visual", [_visual_page(page=1, score=0.5)])
    router = RoutingRetriever(text=text, visual=visual)  # no mode arg → category

    # Definitional query → category=definitional → text-only path
    results = await router.retrieve(Query(text="What is exploration hacking?"))

    assert text.calls == 1
    assert visual.calls == 0
    assert results[0].chunk_id == "paper1::p1::c0"


# --------------------------------------------------------------------------
# Per-query routing_mode override (the demo-UI A/B knob)
# --------------------------------------------------------------------------


def test_query_routing_mode_defaults_to_none() -> None:
    assert Query(text="hello", top_k=5).routing_mode is None


def test_query_routing_mode_accepts_category_and_cascade() -> None:
    assert Query(text="hello", top_k=5, routing_mode="category").routing_mode == "category"
    assert Query(text="hello", top_k=5, routing_mode="cascade").routing_mode == "cascade"


def test_query_routing_mode_rejects_invalid_literal() -> None:
    with pytest.raises(ValidationError):
        Query(text="hello", top_k=5, routing_mode="cascadex")


@pytest.mark.asyncio
async def test_query_routing_mode_cascade_overrides_category_server() -> None:
    """Server wired with mode='category', query asks for cascade: it uses the
    threshold calibrated for the reranker (0.85 for bge-reranker-v2-m3).

    Text score 0.99 (>= 0.85) → cascade returns text-only, visual leg skipped.
    """
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", score=0.99, page=1)])
    visual = _RecordingRetriever("visual", [_visual_page(page=2, score=0.5)])
    router = RoutingRetriever(text=text, visual=visual, default_cascade_threshold=0.85)

    results = await router.retrieve(Query(text="show me the chart", routing_mode="cascade"))

    assert text.calls == 1
    assert visual.calls == 0  # cascade dispatch decided text was confident
    assert results[0].chunk_id == "paper1::p1::c0"


@pytest.mark.asyncio
async def test_query_routing_mode_cascade_invokes_visual_when_uncertain() -> None:
    """Same category server, but low text score → cascade falls through to hybrid."""
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", score=0.3, page=1)])
    visual = _RecordingRetriever("visual", [_visual_page(page=2, score=0.7)])
    router = RoutingRetriever(text=text, visual=visual)  # category mode

    await router.retrieve(Query(text="show me the chart", routing_mode="cascade"))

    assert visual.calls == 1


@pytest.mark.asyncio
async def test_routing_info_captures_category_text_path() -> None:
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", score=0.9)])
    visual = _RecordingRetriever("visual", [])
    router = RoutingRetriever(text=text, visual=visual)

    await router.retrieve(Query(text="explain X"))  # definitional → text

    info = get_last_routing_info()
    assert info is not None
    assert info.mode == "category"
    assert info.path == "text"
    assert info.category == "definitional"
    assert info.forced is False


@pytest.mark.asyncio
async def test_routing_info_captures_cascade_decision() -> None:
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", score=0.99)])
    visual = _RecordingRetriever("visual", [_visual_page(page=2, score=0.5)])
    router = RoutingRetriever(
        text=text, visual=visual, mode="cascade", cascade_confidence_threshold=0.5
    )

    await router.retrieve(Query(text="X?"))

    info = get_last_routing_info()
    assert info is not None
    assert info.mode == "cascade"
    assert info.path == "text"
    assert info.cascade_decision == "confident_text"
    assert info.cascade_top_score == pytest.approx(0.99)
    assert info.cascade_threshold == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_query_routing_mode_category_overrides_cascade_server() -> None:
    """Server wired with mode='cascade', query asks for category — uses regex dispatch.

    Definitional query → text-only path regardless of text-score confidence.
    """
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", score=0.1, page=1)])
    visual = _RecordingRetriever("visual", [_visual_page(page=1, score=0.9)])
    router = RoutingRetriever(
        text=text, visual=visual, mode="cascade", cascade_confidence_threshold=0.5
    )

    # Per-cascade-logic the low text score would fall through to visual. The
    # category override should route definitional → text-only and skip visual.
    await router.retrieve(Query(text="explain the approach", routing_mode="category"))

    assert visual.calls == 0


# ADR 0032: always-hybrid dispatch — fuse both legs, never classify.


class _RecordingClassifier:
    """Test fake: counts classify() calls, returns a fixed category.

    Returns `definitional`, which routes text-only under ADR 0008 dispatch, so
    a visual-leg call proves the classifier was skipped.
    """

    def __init__(self, category: Category = "definitional") -> None:
        self.category = category
        self.calls = 0

    async def classify(self, query: str) -> Category:
        self.calls += 1
        return self.category


@pytest.mark.asyncio
async def test_hybrid_mode_fuses_both_legs_without_classifying() -> None:
    """mode='hybrid' runs both legs on a query the classifier would send to text."""
    classifier = _RecordingClassifier()
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", 0.9)])
    visual = _RecordingRetriever("visual", [_visual_page(2, 0.7)])
    router = RoutingRetriever(
        text=text,
        visual=visual,
        classifier=classifier,  # type: ignore[arg-type]
        mode="hybrid",
    )

    results = await router.retrieve(Query(text="What does the model do?", top_k=5))

    assert classifier.calls == 0
    assert text.calls == 1
    assert visual.calls == 1
    assert {r.chunk_id for r in results} == {"paper1::p1::c0", "paper1::p2::page"}


@pytest.mark.asyncio
async def test_hybrid_mode_reports_itself_in_routing_info() -> None:
    """The /query response names the mode that ran."""
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", 0.9)])
    visual = _RecordingRetriever("visual", [_visual_page(2, 0.7)])
    router = RoutingRetriever(text=text, visual=visual, mode="hybrid")

    await router.retrieve(Query(text="What does the model do?", top_k=5))

    info = get_last_routing_info()
    assert info is not None
    assert info.mode == "hybrid"
    assert info.path == "hybrid"
    assert info.forced is False
    assert info.category is None


@pytest.mark.asyncio
async def test_hybrid_mode_honours_force_route_text() -> None:
    """force_route overrides the server mode."""
    classifier = _RecordingClassifier()
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", 0.9)])
    visual = _RecordingRetriever("visual", [_visual_page(2, 0.7)])
    router = RoutingRetriever(
        text=text,
        visual=visual,
        classifier=classifier,  # type: ignore[arg-type]
        mode="hybrid",
    )

    await router.retrieve(Query(text="Show Figure 3", top_k=5, force_route="text"))

    assert classifier.calls == 0
    assert text.calls == 1
    assert visual.calls == 0
    info = get_last_routing_info()
    assert info is not None
    assert info.mode == "hybrid"
    assert info.path == "text"
    assert info.forced is True


@pytest.mark.asyncio
async def test_hybrid_mode_visual_failure_degrades_to_text() -> None:
    """A raising visual leg returns text-only results (ADR 0008 failure modes)."""
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", 0.9)])
    visual = _FailingRetriever(RuntimeError("CUDA out of memory"))
    router = RoutingRetriever(text=text, visual=visual, mode="hybrid")

    results = await router.retrieve(Query(text="Show Figure 3", top_k=5))

    assert [r.chunk_id for r in results] == ["paper1::p1::c0"]
    info = get_last_routing_info()
    assert info is not None
    assert info.mode == "hybrid"
    assert info.path == "text"
    assert info.visual_failed is True


@pytest.mark.asyncio
async def test_query_routing_mode_hybrid_overrides_category_server() -> None:
    """Query.routing_mode overrides the server mode for one call."""
    classifier = _RecordingClassifier()
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", 0.9)])
    visual = _RecordingRetriever("visual", [_visual_page(2, 0.7)])
    router = RoutingRetriever(
        text=text,
        visual=visual,
        classifier=classifier,  # type: ignore[arg-type]
        mode="category",
    )

    await router.retrieve(Query(text="explain the approach", routing_mode="hybrid"))

    assert classifier.calls == 0
    assert visual.calls == 1


@pytest.mark.asyncio
async def test_hybrid_server_still_serves_forced_visual_only() -> None:
    """force_route='visual' bypasses every mode.

    The logged mode stays the server's.
    """
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", 0.9)])
    visual = _RecordingRetriever("visual", [_visual_page(2, 0.7)])
    router = RoutingRetriever(text=text, visual=visual, mode="hybrid")

    results = await router.retrieve(Query(text="anything", top_k=5, force_route="visual"))

    assert text.calls == 0
    assert [r.chunk_id for r in results] == ["paper1::p2::page"]
    info = get_last_routing_info()
    assert info is not None
    assert info.mode == "hybrid"
    assert info.path == "visual"


@pytest.mark.asyncio
async def test_cascade_without_a_calibrated_threshold_runs_both_legs() -> None:
    """An uncalibrated reranker gives no basis for skipping the visual leg."""
    text = _RecordingRetriever("text", [_text_chunk("paper1::p1::c0", score=0.99, page=1)])
    visual = _RecordingRetriever("visual", [_visual_page(page=2, score=0.5)])
    router = RoutingRetriever(text=text, visual=visual)

    await router.retrieve(Query(text="X?", routing_mode="cascade"))

    info = get_last_routing_info()
    assert visual.calls == 1
    assert info is not None
    assert info.cascade_decision == "uncalibrated_hybrid"
    assert info.cascade_threshold is None


@pytest.mark.asyncio
async def test_cascade_ignores_scores_that_are_not_rerank_scores() -> None:
    """An RRF score (~0.03) must not be read against a rerank threshold."""
    unreranked = _text_chunk("paper1::p1::c0", score=0.99, page=1).model_copy(
        update={"score_kind": "rrf"}
    )
    text = _RecordingRetriever("text", [unreranked])
    visual = _RecordingRetriever("visual", [_visual_page(page=2, score=0.5)])
    router = RoutingRetriever(
        text=text, visual=visual, mode="cascade", cascade_confidence_threshold=0.5
    )

    await router.retrieve(Query(text="X?"))

    assert visual.calls == 1
