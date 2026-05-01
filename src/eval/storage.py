"""Persist EvalRun rows to a SQL backend (Postgres in prod, SQLite in tests).

One row per `EvalRun.run_id`. The full per-query payload is kept as JSON so the
schema doesn't need to evolve every time a new metric is added; aggregates that
matter for trend queries (mean nDCG@5 / recall@10 / MRR over in-corpus queries)
are denormalised into typed columns.

Sync SQLAlchemy 2.0 — the eval CLI is a one-shot script that calls this once
on exit, so sync is fine and avoids dragging in `aiosqlite` for tests.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Engine, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from src.types import EvalRun


class _Base(DeclarativeBase):
    pass


class EvalRunRow(_Base):
    """One row per evaluation run. `config` and `per_query` are JSON-encoded."""

    __tablename__ = "eval_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    finished_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    golden_set_name: Mapped[str] = mapped_column(String, nullable=False)
    golden_set_version: Mapped[str] = mapped_column(String, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    per_query: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    n_queries: Mapped[int] = mapped_column(Integer, nullable=False)
    n_in_corpus_queries: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_ndcg_at_5: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_recall_at_10: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_mrr: Mapped[float | None] = mapped_column(Float, nullable=True)


def _aggregates(run: EvalRun) -> tuple[int, float | None, float | None, float | None]:
    """Macro-mean of retrieval metrics over in-corpus queries (matching report.py)."""
    in_corpus = [q for q in run.per_query if q.category != "out_of_corpus"]
    n = len(in_corpus)
    if n == 0:
        return 0, None, None, None
    return (
        n,
        sum(q.retrieval.ndcg_at_5 for q in in_corpus) / n,
        sum(q.retrieval.recall_at_10 for q in in_corpus) / n,
        sum(q.retrieval.mrr for q in in_corpus) / n,
    )


def to_row(run: EvalRun) -> EvalRunRow:
    """Convert an EvalRun (Pydantic) into a SQLAlchemy row. Pure function, easily testable."""
    n_in_corpus, mean_ndcg5, mean_recall10, mean_mrr = _aggregates(run)
    return EvalRunRow(
        run_id=run.run_id,
        started_at=run.started_at,
        finished_at=run.finished_at,
        golden_set_name=run.golden_set_name,
        golden_set_version=run.golden_set_version,
        config=run.config,
        per_query=[q.model_dump(mode="json") for q in run.per_query],
        n_queries=len(run.per_query),
        n_in_corpus_queries=n_in_corpus,
        mean_ndcg_at_5=mean_ndcg5,
        mean_recall_at_10=mean_recall10,
        mean_mrr=mean_mrr,
    )


def write_eval_run(run: EvalRun, *, engine: Engine) -> None:
    """Idempotently create the table and upsert this run.

    Re-runs with the same `run_id` overwrite — useful when re-judging.
    """
    _Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.merge(to_row(run))
        session.commit()


def make_engine(dsn: str) -> Engine:
    """Construct a SQLAlchemy engine from a DSN. Accepts both `postgresql://` and
    `postgresql+psycopg://`; SQLAlchemy normalises. Use `sqlite:///path` or
    `sqlite:///:memory:` in tests.
    """
    return create_engine(dsn, future=True)
