"""Eval runner: orchestrates a Retriever (and optional Generator + LLM judge) over a GoldenSet."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from src.eval.judges import LLMJudge
from src.eval.metrics_generation import citation_grounding, is_refusal_answer
from src.eval.metrics_retrieval import ndcg_at_k, recall_at_k, reciprocal_rank
from src.observability.logging import get_logger, timed_event
from src.rag.generate import Generator
from src.rag.retrievers.protocol import Retriever
from src.types import (
    EvalRun,
    GenerationMetrics,
    GoldenQuery,
    GoldenSet,
    PerQueryResult,
    Query,
    RetrievalMetrics,
)

_log = get_logger(__name__)


async def evaluate(
    *,
    retriever: Retriever,
    golden_set: GoldenSet,
    generator: Generator | None = None,
    judge: LLMJudge | None = None,
    top_k: int = 10,
    config: dict[str, Any] | None = None,
    paper_id_filter: bool = False,
) -> EvalRun:
    """Run every query in `golden_set` through `retriever` (and `generator` if given).

    If `judge` is provided alongside `generator`, three LLM-as-judge metrics
    (faithfulness, answer_relevance, context_precision) are computed per query.
    `judge` without `generator` only computes `context_precision` (the only
    metric that doesn't need an answer).

    Aggregation is deliberately left to the reporter; this returns raw per-query data.
    """
    with timed_event(
        _log,
        "eval.done",
        golden_set=golden_set.name,
        golden_set_version=golden_set.version,
        n_queries=len(golden_set.queries),
        with_generator=generator is not None,
        with_judge=judge is not None,
    ) as ctx:
        started_at = datetime.now(UTC)
        per_query: list[PerQueryResult] = []
        for query in golden_set.queries:
            per_query.append(
                await _run_one(query, retriever, generator, judge, top_k, paper_id_filter)
            )
        finished_at = datetime.now(UTC)
        ctx["mean_ndcg5"] = (
            sum(r.retrieval.ndcg_at_5 for r in per_query) / len(per_query) if per_query else 0.0
        )
        return EvalRun(
            run_id=uuid4().hex[:12],
            started_at=started_at,
            finished_at=finished_at,
            golden_set_name=golden_set.name,
            golden_set_version=golden_set.version,
            config=config or {},
            per_query=per_query,
        )


async def _run_one(
    query: GoldenQuery,
    retriever: Retriever,
    generator: Generator | None,
    judge: LLMJudge | None,
    top_k: int,
    paper_id_filter: bool = False,
) -> PerQueryResult:
    """Retrieve, optionally generate, optionally judge, compute per-query metrics.

    `paper_id_filter` is the eval-side fairness knob added in the ADR 0009
    follow-up: when True, populate `Query.filters['paper_id']` from the
    golden's `paper_id`. This scopes retrieval to one paper for queries
    whose origin is known — closes the cross-paper bleed pattern observed
    in run ad4fab3bb28d (q9_baselines top-1 was a table from the wrong
    paper). Off by default so non-router eval paths are unchanged.
    """
    started = time.monotonic()
    filters: dict[str, Any] = {}
    if paper_id_filter and query.paper_id:
        filters["paper_id"] = query.paper_id
    rag_query = Query(text=query.text, top_k=top_k, filters=filters)
    retrieved = await retriever.retrieve(rag_query)
    retrieved_chunk_ids = [r.chunk_id for r in retrieved]

    retrieval_metrics = RetrievalMetrics(
        ndcg_at_5=ndcg_at_k(query.relevant_chunk_ids, retrieved_chunk_ids, k=5),
        recall_at_10=recall_at_k(query.relevant_chunk_ids, retrieved_chunk_ids, k=10),
        mrr=reciprocal_rank(query.relevant_chunk_ids, retrieved_chunk_ids),
    )

    answer_text: str | None = None
    cited_chunk_ids: list[str] = []
    tokens_in = 0
    tokens_out = 0
    citation_rate: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    answer_correctness: float | None = None
    faithfulness_std: float | None = None
    answer_relevance_std: float | None = None
    context_precision_std: float | None = None
    answer_correctness_std: float | None = None

    if generator is not None:
        answer = await generator.answer(query.text, retrieved)
        answer_text = answer.text
        cited_chunk_ids = [c.chunk_id for c in answer.citations]
        tokens_in = answer.tokens_in
        tokens_out = answer.tokens_out
        citation_rate = citation_grounding(cited_chunk_ids, retrieved_chunk_ids)

    if judge is not None:
        if answer_text is not None:
            # Deterministic generation scoring for (category, refusal) (B1):
            #
            #   answer is refusal:
            #     OOC      → 1/1  (correct refusal — docs/evals.md convention)
            #     in-corpus → 0/0 (wrong refusal — model gave up on a real query)
            #
            #   answer is NOT a refusal:
            #     OOC      → 0/0 (the model leaked content for an unanswerable
            #                     query — wrong by construction; no LLM judge
            #                     needed. Eliminates the q33-style judge-call
            #                     variance observed in run 196ac0f8786f.)
            #     in-corpus → LLM judge runs (real content evaluation)
            #
            # context_precision still goes through the LLM judge — it scores
            # retrieved chunks, which is orthogonal to whether the answer
            # was a refusal or a leak.
            is_ooc = query.category == "out_of_corpus"
            if is_refusal_answer(answer_text):
                faithfulness = 1.0 if is_ooc else 0.0
                answer_relevance = 1.0 if is_ooc else 0.0
                # answer_correctness is only defined when ground-truth facts
                # exist (in-corpus queries with expected_facts). A wrong
                # refusal on such a query covers zero facts; OOC has no
                # expected_facts so the metric stays None.
                if not is_ooc and query.expected_facts:
                    answer_correctness = 0.0
            elif is_ooc:
                faithfulness = 0.0
                answer_relevance = 0.0
            else:
                faith_out = await judge.faithfulness(
                    query=query.text, answer=answer_text, retrieved=retrieved
                )
                ans_out = await judge.answer_relevance(query=query.text, answer=answer_text)
                faithfulness = faith_out.score
                answer_relevance = ans_out.score
                # Std fields surface multi-seed variance (B2). For single-seed
                # the std is 0.0; the deterministic-override paths above leave
                # these fields None to signal "no measurable variance, by
                # construction".
                faithfulness_std = faith_out.score_std
                answer_relevance_std = ans_out.score_std
                # answer_correctness vs expected_facts — ADR 0019's
                # chunk-id-robust scoreboard. Skipped when the judge wasn't
                # configured with the prompt (older callers / cheap-eval
                # paths) or the query has no ground-truth facts.
                # getattr defends against older judge stubs / mocks that
                # predate the answer_correctness extension.
                if getattr(judge, "has_answer_correctness", False) and query.expected_facts:
                    ac_out = await judge.answer_correctness(
                        query=query.text,
                        answer=answer_text,
                        expected_facts=query.expected_facts,
                    )
                    answer_correctness = ac_out.score
                    answer_correctness_std = ac_out.score_std
        ctx_prec_out = await judge.context_precision(query=query.text, retrieved=retrieved)
        context_precision = ctx_prec_out.score
        context_precision_std = ctx_prec_out.score_std

    generation_metrics: GenerationMetrics | None = None
    if generator is not None or judge is not None:
        generation_metrics = GenerationMetrics(
            citation_rate=citation_rate,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            answer_correctness=answer_correctness,
            faithfulness_std=faithfulness_std,
            answer_relevance_std=answer_relevance_std,
            context_precision_std=context_precision_std,
            answer_correctness_std=answer_correctness_std,
        )

    return PerQueryResult(
        query_id=query.query_id,
        category=query.category,
        text=query.text,
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieval=retrieval_metrics,
        generation=generation_metrics,
        answer_text=answer_text,
        cited_chunk_ids=cited_chunk_ids,
        latency_ms=int((time.monotonic() - started) * 1000),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )
