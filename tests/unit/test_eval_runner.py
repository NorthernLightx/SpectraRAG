"""Eval runner: orchestrates Retriever (+ optional Generator) over a GoldenSet."""

from __future__ import annotations

from typing import Any

import pytest

from src.eval.runner import evaluate
from src.types import (
    Answer,
    Citation,
    GoldenQuery,
    GoldenSet,
    Query,
    RetrievalResult,
)


class _CannedRetriever:
    """Returns fixed RetrievalResults regardless of query."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        return self._results[: query.top_k]


class _CannedGenerator:
    """Returns a fixed Answer regardless of input."""

    def __init__(self, answer: Answer) -> None:
        self._answer = answer

    async def answer(self, query: str, retrieved: list[RetrievalResult]) -> Answer:
        return self._answer


def _retrieval(cid: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=cid,
        paper_id="p1",
        score=0.9,
        text=f"text of {cid}",
        page_numbers=[1],
        source="pipeline",
    )


def _golden(
    query_id: str, category: str = "factual", relevant: list[str] | None = None
) -> GoldenQuery:
    return GoldenQuery(
        query_id=query_id,
        text=f"q-{query_id}",
        paper_id="p1",
        category=category,  # type: ignore[arg-type]
        relevant_chunk_ids=relevant or [],
    )


async def test_evaluate_retriever_only_no_generation() -> None:
    retriever = _CannedRetriever([_retrieval("c1"), _retrieval("c2")])
    gs = GoldenSet(
        name="tiny",
        version="v1",
        queries=[_golden("q1", relevant=["c1"])],
    )
    run = await evaluate(retriever=retriever, golden_set=gs)

    assert len(run.per_query) == 1
    pq = run.per_query[0]
    assert pq.retrieved_chunk_ids == ["c1", "c2"]
    assert pq.retrieval.ndcg_at_5 == pytest.approx(1.0)
    assert pq.retrieval.recall_at_10 == pytest.approx(1.0)
    assert pq.retrieval.mrr == pytest.approx(1.0)
    assert pq.generation is None
    assert pq.answer_text is None


async def test_evaluate_with_generator_populates_generation_metrics() -> None:
    retriever = _CannedRetriever([_retrieval("c1"), _retrieval("c2")])
    fake_answer = Answer(
        text="From [c1] we conclude X.",
        citations=[Citation(chunk_id="c1", paper_id="p1", page_numbers=[1])],
        model="anthropic/claude-3.5-sonnet",
        prompt_version="v1",
        latency_ms=100,
        tokens_in=42,
        tokens_out=12,
    )
    gs = GoldenSet(
        name="tiny",
        version="v1",
        queries=[_golden("q1", relevant=["c1"])],
    )

    run = await evaluate(
        retriever=retriever, golden_set=gs, generator=_CannedGenerator(fake_answer)
    )

    pq = run.per_query[0]
    assert pq.generation is not None
    assert pq.generation.citation_rate == pytest.approx(1.0)
    assert pq.cited_chunk_ids == ["c1"]
    assert pq.tokens_in == 42 and pq.tokens_out == 12
    assert pq.answer_text == "From [c1] we conclude X."


async def test_evaluate_out_of_corpus_query_yields_zero_metrics() -> None:
    retriever = _CannedRetriever([_retrieval("c1"), _retrieval("c2")])
    gs = GoldenSet(
        name="tiny",
        version="v1",
        queries=[_golden("q_oc", category="out_of_corpus", relevant=[])],
    )
    run = await evaluate(retriever=retriever, golden_set=gs)

    pq = run.per_query[0]
    assert pq.retrieval.ndcg_at_5 == 0.0
    assert pq.retrieval.recall_at_10 == 0.0
    assert pq.retrieval.mrr == 0.0


async def test_evaluate_records_config_block() -> None:
    retriever = _CannedRetriever([])
    gs = GoldenSet(name="tiny", version="v1", queries=[])
    cfg: dict[str, Any] = {"retriever": "pipeline", "rerank": True, "top_k": 10}
    run = await evaluate(retriever=retriever, golden_set=gs, config=cfg)
    assert run.config == cfg
    assert run.per_query == []
