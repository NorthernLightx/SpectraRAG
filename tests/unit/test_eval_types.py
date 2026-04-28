"""Eval Pydantic types: instantiate, JSON round-trip, category validation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.types import (
    EvalRun,
    GenerationMetrics,
    GoldenQuery,
    GoldenSet,
    PerQueryResult,
    RetrievalMetrics,
)


def test_golden_query_minimal() -> None:
    q = GoldenQuery(
        query_id="q1",
        text="What is X?",
        paper_id="p1",
        category="factual",
        relevant_chunk_ids=["p1::p1::c0"],
        expected_facts=["X is the answer."],
    )
    assert q.category == "factual"
    assert q.relevant_pages == []


def test_golden_query_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        GoldenQuery(
            query_id="q1",
            text="?",
            paper_id="p1",
            category="not-a-category",
        )


def test_golden_set_round_trip() -> None:
    gs = GoldenSet(
        name="phase1",
        version="v1",
        queries=[
            GoldenQuery(query_id="q1", text="?", paper_id="p1", category="factual"),
            GoldenQuery(query_id="q2", text="?", paper_id="p1", category="out_of_corpus"),
        ],
    )
    payload = gs.model_dump_json()
    restored = GoldenSet.model_validate_json(payload)
    assert restored == gs


def test_eval_run_round_trip() -> None:
    now = datetime.now(UTC)
    run = EvalRun(
        run_id="r1",
        started_at=now,
        finished_at=now,
        golden_set_name="phase1",
        golden_set_version="v1",
        config={"retriever": "pipeline", "rerank": True},
        per_query=[
            PerQueryResult(
                query_id="q1",
                category="factual",
                text="?",
                retrieved_chunk_ids=["p1::p1::c0"],
                retrieval=RetrievalMetrics(ndcg_at_5=1.0, recall_at_10=1.0, mrr=1.0),
                generation=GenerationMetrics(citation_rate=1.0),
                latency_ms=120,
                tokens_in=80,
                tokens_out=24,
                answer_text="The answer is X [p1::p1::c0].",
                cited_chunk_ids=["p1::p1::c0"],
            )
        ],
    )
    payload = run.model_dump_json()
    restored = EvalRun.model_validate_json(payload)
    assert restored.run_id == "r1"
    assert restored.per_query[0].retrieval.ndcg_at_5 == 1.0


def test_per_query_result_rejects_negative_latency() -> None:
    with pytest.raises(ValidationError):
        PerQueryResult(
            query_id="q1",
            category="factual",
            text="?",
            retrieved_chunk_ids=[],
            retrieval=RetrievalMetrics(ndcg_at_5=0.0, recall_at_10=0.0, mrr=0.0),
            latency_ms=-1,
        )
