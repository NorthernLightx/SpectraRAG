"""Eval runner: orchestrates Retriever (+ optional Generator) over a GoldenSet."""

from __future__ import annotations

from typing import Any, cast

import pytest

from src.eval.judges import JudgeOutput, LLMJudge
from src.eval.runner import evaluate
from src.rag.generate import Generator
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
        category=category,
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
        retriever=retriever,
        golden_set=gs,
        generator=cast(Generator, _CannedGenerator(fake_answer)),
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


# Deterministic refusal handling per docs/evals.md OOC convention.
# The LLM judge was empirically unreliable on refusal answers (run
# c92f3f1bee19: 5 of 6 OOC refusals scored 0.0 on faithfulness despite
# the documented convention that they should score 1.0). The runner now
# detects refusal sentinels and short-circuits faithfulness +
# answer_relevance scoring. context_precision still goes through the
# judge — it scores retrieved chunks, orthogonal to refusal status.


class _RecordingJudge:
    """Records every judge call. Returns 0.5 by default — visibly different
    from the deterministic 1.0/0.0 the runner produces for refusals, so
    accidental fallthrough surfaces as a test failure."""

    def __init__(self) -> None:
        self.faithfulness_calls = 0
        self.answer_relevance_calls = 0
        self.context_precision_calls = 0

    async def faithfulness(
        self, *, query: str, answer: str, retrieved: list[RetrievalResult]
    ) -> JudgeOutput:
        self.faithfulness_calls += 1
        return JudgeOutput(score=0.5, rationale="stub", model="stub", prompt_version="stub-v0")

    async def answer_relevance(self, *, query: str, answer: str) -> JudgeOutput:
        self.answer_relevance_calls += 1
        return JudgeOutput(score=0.5, rationale="stub", model="stub", prompt_version="stub-v0")

    async def context_precision(
        self, *, query: str, retrieved: list[RetrievalResult]
    ) -> JudgeOutput:
        self.context_precision_calls += 1
        return JudgeOutput(score=0.5, rationale="stub", model="stub", prompt_version="stub-v0")


def _refusal_answer() -> Answer:
    return Answer(
        text="Not stated in the provided context.  \nCitations: None",
        citations=[],
        model="stub",
        prompt_version="stub-v0",
        latency_ms=0,
        tokens_in=10,
        tokens_out=12,
    )


def _substantive_answer() -> Answer:
    return Answer(
        text="From [c1] we conclude X.",
        citations=[Citation(chunk_id="c1", paper_id="p1", page_numbers=[1])],
        model="stub",
        prompt_version="stub-v0",
        latency_ms=0,
        tokens_in=42,
        tokens_out=12,
    )


async def test_refusal_on_ooc_scores_one_without_llm_judge() -> None:
    retriever = _CannedRetriever([_retrieval("c1")])
    judge = _RecordingJudge()
    gs = GoldenSet(
        name="tiny",
        version="v1",
        queries=[_golden("q_oc", category="out_of_corpus", relevant=[])],
    )

    run = await evaluate(
        retriever=retriever,
        golden_set=gs,
        generator=cast(Generator, _CannedGenerator(_refusal_answer())),
        judge=cast(LLMJudge, judge),
    )

    pq = run.per_query[0]
    assert pq.generation is not None
    assert pq.generation.faithfulness == pytest.approx(1.0)
    assert pq.generation.answer_relevance == pytest.approx(1.0)
    assert pq.generation.context_precision == pytest.approx(0.5)
    assert judge.faithfulness_calls == 0
    assert judge.answer_relevance_calls == 0
    assert judge.context_precision_calls == 1


async def test_refusal_on_in_corpus_scores_zero_without_llm_judge() -> None:
    retriever = _CannedRetriever([_retrieval("c1")])
    judge = _RecordingJudge()
    gs = GoldenSet(
        name="tiny",
        version="v1",
        queries=[_golden("q1", category="factual", relevant=["c1"])],
    )

    run = await evaluate(
        retriever=retriever,
        golden_set=gs,
        generator=cast(Generator, _CannedGenerator(_refusal_answer())),
        judge=cast(LLMJudge, judge),
    )

    pq = run.per_query[0]
    assert pq.generation is not None
    assert pq.generation.faithfulness == pytest.approx(0.0)
    assert pq.generation.answer_relevance == pytest.approx(0.0)
    assert judge.faithfulness_calls == 0
    assert judge.answer_relevance_calls == 0


async def test_substantive_answer_uses_llm_judge() -> None:
    retriever = _CannedRetriever([_retrieval("c1")])
    judge = _RecordingJudge()
    gs = GoldenSet(
        name="tiny",
        version="v1",
        queries=[_golden("q1", category="factual", relevant=["c1"])],
    )

    run = await evaluate(
        retriever=retriever,
        golden_set=gs,
        generator=cast(Generator, _CannedGenerator(_substantive_answer())),
        judge=cast(LLMJudge, judge),
    )

    pq = run.per_query[0]
    assert pq.generation is not None
    assert pq.generation.faithfulness == pytest.approx(0.5)
    assert pq.generation.answer_relevance == pytest.approx(0.5)
    assert pq.generation.context_precision == pytest.approx(0.5)
    assert judge.faithfulness_calls == 1
    assert judge.answer_relevance_calls == 1
    assert judge.context_precision_calls == 1


async def test_non_refusal_on_ooc_scores_zero_without_llm_judge() -> None:
    """B1 (deterministic OOC scoring): a non-refusal answer to an OOC query
    is wrong by construction — no LLM judge needed. Eliminates the
    q33-style judge-call variance observed in run 196ac0f8786f."""
    retriever = _CannedRetriever([_retrieval("c1")])
    judge = _RecordingJudge()
    leaked_answer = Answer(
        text="The hyperparameters are learning rate 1e-5, 15000 steps, ...",
        citations=[Citation(chunk_id="c1", paper_id="p1", page_numbers=[1])],
        model="stub",
        prompt_version="stub-v0",
        latency_ms=0,
        tokens_in=42,
        tokens_out=12,
    )
    gs = GoldenSet(
        name="tiny",
        version="v1",
        queries=[_golden("q_oc", category="out_of_corpus", relevant=[])],
    )

    run = await evaluate(
        retriever=retriever,
        golden_set=gs,
        generator=cast(Generator, _CannedGenerator(leaked_answer)),
        judge=cast(LLMJudge, judge),
    )

    pq = run.per_query[0]
    assert pq.generation is not None
    assert pq.generation.faithfulness == pytest.approx(0.0)
    assert pq.generation.answer_relevance == pytest.approx(0.0)
    # context_precision still goes through the judge — orthogonal to whether
    # the answer was a leak.
    assert pq.generation.context_precision == pytest.approx(0.5)
    # Critical: faithfulness/answer_relevance LLM judge calls are SKIPPED.
    assert judge.faithfulness_calls == 0
    assert judge.answer_relevance_calls == 0
    assert judge.context_precision_calls == 1


async def test_refusal_on_ooc_with_alternate_gate_phrase() -> None:
    retriever = _CannedRetriever([_retrieval("c1")])
    judge = _RecordingJudge()
    gate_refusal = Answer(
        text="I cannot answer this question from the provided corpus.",
        citations=[],
        model="refusal-gate",
        prompt_version="refusal-v1",
        latency_ms=0,
        tokens_in=0,
        tokens_out=0,
    )
    gs = GoldenSet(
        name="tiny",
        version="v1",
        queries=[_golden("q_oc", category="out_of_corpus", relevant=[])],
    )

    run = await evaluate(
        retriever=retriever,
        golden_set=gs,
        generator=cast(Generator, _CannedGenerator(gate_refusal)),
        judge=cast(LLMJudge, judge),
    )

    pq = run.per_query[0]
    assert pq.generation is not None
    assert pq.generation.faithfulness == pytest.approx(1.0)
    assert pq.generation.answer_relevance == pytest.approx(1.0)
    assert judge.faithfulness_calls == 0
    assert judge.answer_relevance_calls == 0


# ADR 0009 follow-up: paper_id_filter populates Query.filters['paper_id']
# from GoldenQuery.paper_id, scoping retrieval to a single paper for
# queries whose origin is known.


class _FilterCapturingRetriever:
    """Records the filters dict each query was retrieved with."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.captured_filters: list[dict[str, Any]] = []

    async def retrieve(self, query: Query) -> list[RetrievalResult]:
        self.captured_filters.append(dict(query.filters))
        return self._results[: query.top_k]


async def test_paper_id_filter_off_passes_empty_filters() -> None:
    """Default behavior: no filter populated."""
    captor = _FilterCapturingRetriever([_retrieval("c1")])
    gs = GoldenSet(
        name="tiny",
        version="v1",
        queries=[_golden("q1", category="factual", relevant=["c1"])],
    )
    await evaluate(retriever=captor, golden_set=gs, paper_id_filter=False)
    assert captor.captured_filters == [{}]


async def test_paper_id_filter_on_populates_from_golden() -> None:
    """When the flag is on, filters['paper_id'] = golden.paper_id."""
    captor = _FilterCapturingRetriever([_retrieval("c1")])
    gs = GoldenSet(
        name="tiny",
        version="v1",
        queries=[_golden("q1", category="factual", relevant=["c1"])],
    )
    await evaluate(retriever=captor, golden_set=gs, paper_id_filter=True)
    assert captor.captured_filters == [{"paper_id": "p1"}]


async def test_paper_id_filter_on_works_for_ooc_queries_too() -> None:
    """OOC queries also have a paper_id label; filter applies uniformly. Result
    is fine — OOC retrieval is 0 by construction regardless of filter."""
    captor = _FilterCapturingRetriever([])
    gs = GoldenSet(
        name="tiny",
        version="v1",
        queries=[_golden("q_oc", category="out_of_corpus", relevant=[])],
    )
    await evaluate(retriever=captor, golden_set=gs, paper_id_filter=True)
    assert captor.captured_filters == [{"paper_id": "p1"}]
