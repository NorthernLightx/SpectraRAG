"""Storage round-trip: write an EvalRun, read it back, verify aggregates + JSON payload."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.eval.storage import EvalRunRow, _aggregates, make_engine, to_row, write_eval_run
from src.types import EvalRun, GenerationMetrics, PerQueryResult, RetrievalMetrics


def _per_query(
    query_id: str, category: str, ndcg5: float, recall10: float, mrr: float
) -> PerQueryResult:
    return PerQueryResult(
        query_id=query_id,
        category=category,
        text="dummy",
        retrieved_chunk_ids=["c1", "c2"],
        retrieval=RetrievalMetrics(ndcg_at_5=ndcg5, recall_at_10=recall10, mrr=mrr),
        generation=GenerationMetrics(faithfulness=0.9),
        answer_text="answer",
        cited_chunk_ids=["c1"],
        latency_ms=200,
        tokens_in=10,
        tokens_out=5,
    )


def _eval_run(run_id: str = "abc123") -> EvalRun:
    return EvalRun(
        run_id=run_id,
        started_at=datetime(2026, 5, 1, 9, 0, 0, tzinfo=UTC),
        finished_at=datetime(2026, 5, 1, 9, 1, 0, tzinfo=UTC),
        golden_set_name="phase1-text-baseline",
        golden_set_version="v1",
        config={"retriever": "pipeline", "top_k": 10},
        per_query=[
            _per_query("q1", "factual", 1.0, 1.0, 1.0),
            _per_query("q2", "factual", 0.5, 1.0, 0.5),
            _per_query("q3", "out_of_corpus", 0.0, 0.0, 0.0),
        ],
    )


def test_aggregates_excludes_out_of_corpus() -> None:
    """In-corpus mean over 2 of the 3 queries; OOC excluded."""
    run = _eval_run()
    n, ndcg5, recall10, mrr = _aggregates(run)
    assert n == 2
    assert ndcg5 == 0.75
    assert recall10 == 1.0
    assert mrr == 0.75


def test_aggregates_returns_none_when_all_out_of_corpus() -> None:
    run = _eval_run()
    run.per_query = [_per_query("q3", "out_of_corpus", 0.0, 0.0, 0.0)]
    n, ndcg5, recall10, mrr = _aggregates(run)
    assert (n, ndcg5, recall10, mrr) == (0, None, None, None)


def test_to_row_serialises_per_query_to_json() -> None:
    run = _eval_run()
    row = to_row(run)
    assert row.run_id == "abc123"
    assert row.n_queries == 3
    assert row.n_in_corpus_queries == 2
    assert row.mean_ndcg_at_5 == 0.75
    assert isinstance(row.per_query, list)
    assert row.per_query[0]["query_id"] == "q1"
    assert row.per_query[0]["retrieval"]["ndcg_at_5"] == 1.0


def test_write_eval_run_roundtrip_via_sqlite() -> None:
    """End-to-end: write to in-memory SQLite, read back, verify."""
    engine = make_engine("sqlite:///:memory:")
    run = _eval_run()
    write_eval_run(run, engine=engine)

    with Session(engine) as session:
        rows = session.execute(select(EvalRunRow)).scalars().all()
    assert len(rows) == 1
    [row] = rows
    assert row.run_id == "abc123"
    assert row.golden_set_version == "v1"
    assert row.config == {"retriever": "pipeline", "top_k": 10}
    assert row.mean_ndcg_at_5 == 0.75
    assert row.per_query[1]["query_id"] == "q2"


def test_write_eval_run_idempotent_on_same_run_id() -> None:
    """Same run_id → upsert (merge), not duplicate."""
    engine = make_engine("sqlite:///:memory:")
    run = _eval_run("dup-id")
    write_eval_run(run, engine=engine)
    # Mutate and re-write
    run.config = {"retriever": "pipeline", "top_k": 20}
    write_eval_run(run, engine=engine)

    with Session(engine) as session:
        rows = session.execute(select(EvalRunRow)).scalars().all()
    assert len(rows) == 1
    assert rows[0].config == {"retriever": "pipeline", "top_k": 20}
