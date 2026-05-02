"""Offline hybrid (text + visual) retrieval fusion at page granularity.

Pure-function unit tests for the helpers in `scripts/eval_hybrid.py`. The
script itself is just orchestration around these helpers + reuse of
`src.rag.hybrid.reciprocal_rank_fusion` and `src.eval.metrics_retrieval`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from scripts.eval_hybrid import (
    _chunk_id_to_page_id,
    _chunks_to_pages,
    _fuse_pages,
    _hybrid_per_query,
    _relevant_pages_for,
    _validate_runs_compatible,
)
from src.types import (
    EvalRun,
    GoldenQuery,
    PerQueryResult,
    RetrievalMetrics,
)

# ---------- chunk → page id normalisation ----------


def test_chunk_id_to_page_id_normalises_text_chunk_id() -> None:
    assert _chunk_id_to_page_id("2604.22753v1::p5::c24") == "2604.22753v1::p5::page"


def test_chunk_id_to_page_id_idempotent_on_visual_page_id() -> None:
    assert _chunk_id_to_page_id("2604.22753v1::p5::page") == "2604.22753v1::p5::page"


def test_chunk_id_to_page_id_preserves_paper_with_dots() -> None:
    assert _chunk_id_to_page_id("2604.28180v1::p12::c47") == "2604.28180v1::p12::page"


def test_chunk_id_to_page_id_raises_on_malformed() -> None:
    with pytest.raises(ValueError):
        _chunk_id_to_page_id("not-an-id")


def test_chunk_id_to_page_id_raises_on_missing_page_segment() -> None:
    with pytest.raises(ValueError):
        _chunk_id_to_page_id("paper::notpage::c1")


# ---------- chunks → pages (dedup preserving rank) ----------


def test_chunks_to_pages_dedupes_preserving_first_occurrence_order() -> None:
    chunks = [
        "paper::p5::c24",
        "paper::p5::c23",  # same page, lower rank — should drop
        "paper::p25::c89",
        "paper::p23::c84",
        "paper::p5::c20",  # same page again — drop
    ]
    assert _chunks_to_pages(chunks) == [
        "paper::p5::page",
        "paper::p25::page",
        "paper::p23::page",
    ]


def test_chunks_to_pages_handles_empty_input() -> None:
    assert _chunks_to_pages([]) == []


def test_chunks_to_pages_normalises_visual_page_ids() -> None:
    """Visual run already emits page-ids; pass through unchanged."""
    pages = ["paper::p1::page", "paper::p2::page", "paper::p1::page"]
    assert _chunks_to_pages(pages) == ["paper::p1::page", "paper::p2::page"]


# ---------- golden query → relevant pages ----------


def _golden(
    *,
    qid: str = "q1",
    paper_id: str = "paper",
    chunks: list[str] | None = None,
    pages: list[int] | None = None,
    category: str = "factual",
) -> GoldenQuery:
    return GoldenQuery(
        query_id=qid,
        text="t",
        paper_id=paper_id,
        category=category,
        relevant_chunk_ids=chunks or [],
        relevant_pages=pages or [],
    )


def test_relevant_pages_for_uses_relevant_pages_field_when_present() -> None:
    q = _golden(paper_id="paper", pages=[3, 7])
    assert _relevant_pages_for(q) == ["paper::p3::page", "paper::p7::page"]


def test_relevant_pages_for_falls_back_to_chunk_ids_when_pages_empty() -> None:
    q = _golden(chunks=["paper::p5::c1", "paper::p7::c2", "paper::p5::c3"])
    # de-duped, sorted by page number for determinism
    assert _relevant_pages_for(q) == ["paper::p5::page", "paper::p7::page"]


def test_relevant_pages_for_empty_when_ooc() -> None:
    q = _golden(category="out_of_corpus")
    assert _relevant_pages_for(q) == []


# ---------- RRF page fusion ----------


def test_fuse_pages_promotes_consensus_top() -> None:
    """A page ranked top in both lists should outrank pages in only one."""
    text = ["A", "B", "C"]
    visual = ["A", "D", "E"]
    fused = _fuse_pages(text, visual, rrf_k=60, top_k=5)
    assert fused[0] == "A"


def test_fuse_pages_returns_top_k_only() -> None:
    text = [f"t{i}" for i in range(20)]
    visual = [f"v{i}" for i in range(20)]
    fused = _fuse_pages(text, visual, rrf_k=60, top_k=10)
    assert len(fused) == 10


def test_fuse_pages_handles_empty_lists() -> None:
    assert _fuse_pages([], [], rrf_k=60, top_k=5) == []


def test_fuse_pages_with_only_text_preserves_order() -> None:
    text = ["A", "B", "C"]
    fused = _fuse_pages(text, [], rrf_k=60, top_k=10)
    assert fused == ["A", "B", "C"]


# ---------- run-compatibility validation ----------


def _empty_run(name: str = "phase1-text-baseline", version: str = "v2") -> EvalRun:
    now = datetime.now(UTC)
    return EvalRun(
        run_id="x",
        started_at=now,
        finished_at=now,
        golden_set_name=name,
        golden_set_version=version,
        config={},
        per_query=[],
    )


def test_validate_runs_compatible_passes_when_same_golden() -> None:
    _validate_runs_compatible(_empty_run(), _empty_run())  # no raise


def test_validate_runs_compatible_raises_on_different_golden_name() -> None:
    with pytest.raises(ValueError, match="golden_set"):
        _validate_runs_compatible(_empty_run(name="A"), _empty_run(name="B"))


def test_validate_runs_compatible_raises_on_different_version() -> None:
    with pytest.raises(ValueError, match="version"):
        _validate_runs_compatible(_empty_run(version="v1"), _empty_run(version="v2"))


# ---------- per-query orchestration ----------


def _per_query_result(
    qid: str,
    chunks: list[str],
    *,
    category: str = "factual",
    text: str = "t",
) -> PerQueryResult:
    return PerQueryResult(
        query_id=qid,
        category=category,
        text=text,
        retrieved_chunk_ids=chunks,
        retrieval=RetrievalMetrics(ndcg_at_5=0.0, recall_at_10=0.0, mrr=0.0),
        latency_ms=1,
    )


def test_hybrid_per_query_perfect_when_relevant_at_top_in_both_lists() -> None:
    """Both retrievers rank the relevant page first → fused top is also that page."""
    golden = _golden(qid="q1", chunks=["paper::p5::c1"])
    text = _per_query_result("q1", ["paper::p5::c1", "paper::p5::c2", "paper::p9::c3"])
    visual = _per_query_result("q1", ["paper::p5::page", "paper::p7::page"])

    result = _hybrid_per_query(golden, text=text, visual=visual, rrf_k=60, top_k=10)

    assert result.query_id == "q1"
    assert result.retrieved_chunk_ids[0] == "paper::p5::page"
    assert result.retrieval.ndcg_at_5 == pytest.approx(1.0)
    assert result.retrieval.recall_at_10 == pytest.approx(1.0)
    assert result.retrieval.mrr == pytest.approx(1.0)


def test_hybrid_per_query_recovers_when_only_visual_has_relevant_in_top() -> None:
    """Text misses; visual finds → fused recall@10 should be 1.0 thanks to visual."""
    golden = _golden(qid="q2", chunks=["paper::p9::c5"])
    text = _per_query_result("q2", ["paper::p1::c1", "paper::p2::c2"])
    visual = _per_query_result("q2", ["paper::p9::page", "paper::p10::page"])

    result = _hybrid_per_query(golden, text=text, visual=visual, rrf_k=60, top_k=10)

    assert "paper::p9::page" in result.retrieved_chunk_ids
    assert result.retrieval.recall_at_10 == pytest.approx(1.0)


def test_hybrid_per_query_zero_metrics_for_ooc() -> None:
    """OOC query has no relevant pages — metrics return 0 by metric convention."""
    golden = _golden(qid="q5", category="out_of_corpus")
    text = _per_query_result("q5", ["paper::p1::c1"], category="out_of_corpus")
    visual = _per_query_result("q5", ["paper::p1::page"], category="out_of_corpus")

    result = _hybrid_per_query(golden, text=text, visual=visual, rrf_k=60, top_k=10)

    assert result.category == "out_of_corpus"
    assert result.retrieval.ndcg_at_5 == 0.0
    assert result.retrieval.recall_at_10 == 0.0
    assert result.retrieval.mrr == 0.0


def test_hybrid_per_query_carries_category_and_text_from_golden() -> None:
    golden = _golden(qid="qX", category="multi_hop", chunks=["paper::p2::c1"])
    text = _per_query_result("qX", ["paper::p2::c1"], category="multi_hop")
    visual = _per_query_result("qX", ["paper::p2::page"], category="multi_hop")
    result = _hybrid_per_query(golden, text=text, visual=visual, rrf_k=60, top_k=10)
    assert result.category == "multi_hop"
