"""Validate that shared Pydantic types instantiate and round-trip via JSON."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from src.types import (
    Answer,
    Chunk,
    Citation,
    Context,
    Figure,
    Page,
    Paper,
    Query,
    RankedChunk,
    RetrievalResult,
    Table,
)


def test_paper_minimal_fields() -> None:
    paper = Paper(
        paper_id="p1",
        arxiv_id="2401.00001",
        title="Test paper",
        abstract="Abstract.",
        pdf_path=Path("data/papers/p1.pdf"),
        authors=["A. Author"],
    )
    assert paper.paper_id == "p1"
    assert paper.model_dump_json()


def test_paper_requires_paper_id() -> None:
    with pytest.raises(ValidationError):
        Paper(  # type: ignore[call-arg]
            arxiv_id="2401.00001",
            title="Test",
            pdf_path=Path("x.pdf"),
            authors=[],
        )


def test_chunk_round_trip() -> None:
    chunk = Chunk(
        chunk_id="c1",
        paper_id="p1",
        page_numbers=[1, 2],
        text="Some text.",
        section="Introduction",
    )
    payload = chunk.model_dump_json()
    restored = Chunk.model_validate_json(payload)
    assert restored == chunk


def test_query_defaults() -> None:
    q = Query(text="What is X?")
    assert q.top_k == 5
    assert q.filters == {}


def test_retrieval_result_source_constrained() -> None:
    RetrievalResult(
        chunk_id="c1",
        paper_id="p1",
        score=0.9,
        text="snippet",
        page_numbers=[1],
        source="pipeline",
    )
    with pytest.raises(ValidationError):
        RetrievalResult(
            chunk_id="c1",
            paper_id="p1",
            score=0.9,
            text="snippet",
            page_numbers=[1],
            source="not-a-real-source",
        )


def test_ranked_chunk_carries_chunk() -> None:
    chunk = Chunk(chunk_id="c1", paper_id="p1", page_numbers=[1], text="t")
    ranked = RankedChunk(chunk=chunk, score=0.8, rank=1)
    assert ranked.chunk.chunk_id == "c1"
    assert ranked.rerank_score is None


def test_context_aggregates_chunks() -> None:
    chunk = Chunk(chunk_id="c1", paper_id="p1", page_numbers=[1], text="t")
    ctx = Context(
        query="q",
        chunks=[RankedChunk(chunk=chunk, score=0.8, rank=1)],
        token_count=42,
    )
    assert ctx.token_count == 42
    assert len(ctx.chunks) == 1


def test_answer_with_citation() -> None:
    cit = Citation(chunk_id="c1", paper_id="p1", page_numbers=[1])
    ans = Answer(
        text="Answer.",
        citations=[cit],
        model="anthropic/claude-3.5-sonnet",
        latency_ms=1234,
        tokens_in=100,
        tokens_out=50,
    )
    assert ans.tokens_in + ans.tokens_out == 150


def test_figure_and_table_models() -> None:
    fig = Figure(
        figure_id="f1",
        paper_id="p1",
        page_number=3,
        caption="Figure 1: ...",
        image_path=Path("data/figures/p1_f1.png"),
    )
    assert fig.vlm_caption is None

    tbl = Table(table_id="t1", paper_id="p1", page_number=4, markdown="| a |\n|---|\n| 1 |")
    assert "|" in tbl.markdown


def test_page_model() -> None:
    page = Page(paper_id="p1", page_number=1, text="hello")
    assert page.image_path is None
