"""Eval runner: orchestrates a Retriever (and optional Generator) over a GoldenSet."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from src.eval.metrics_generation import citation_grounding
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
    top_k: int = 10,
    config: dict[str, Any] | None = None,
) -> EvalRun:
    """Run every query in `golden_set` through `retriever` (and `generator` if given).

    Aggregation is deliberately left to the reporter; this returns raw per-query data.
    """
    with timed_event(
        _log,
        "eval.done",
        golden_set=golden_set.name,
        golden_set_version=golden_set.version,
        n_queries=len(golden_set.queries),
        with_generator=generator is not None,
    ) as ctx:
        started_at = datetime.now(UTC)
        per_query: list[PerQueryResult] = []
        for query in golden_set.queries:
            per_query.append(await _run_one(query, retriever, generator, top_k))
        finished_at = datetime.now(UTC)
        ctx["mean_ndcg5"] = (
            sum(r.retrieval.ndcg_at_5 for r in per_query) / len(per_query)
            if per_query
            else 0.0
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
    top_k: int,
) -> PerQueryResult:
    """Retrieve, optionally generate, compute per-query metrics."""
    started = time.monotonic()
    rag_query = Query(text=query.text, top_k=top_k)
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
    generation_metrics: GenerationMetrics | None = None

    if generator is not None:
        answer = await generator.answer(query.text, retrieved)
        answer_text = answer.text
        cited_chunk_ids = [c.chunk_id for c in answer.citations]
        tokens_in = answer.tokens_in
        tokens_out = answer.tokens_out
        generation_metrics = GenerationMetrics(
            citation_rate=citation_grounding(cited_chunk_ids, retrieved_chunk_ids),
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
