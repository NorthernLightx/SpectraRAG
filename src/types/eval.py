"""Evaluation-stage Pydantic models. Imported as `from src.types.eval import ...`."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

QueryCategory = Literal[
    "factual",
    "multi_hop",
    "figure",
    "table",
    "equation",
    "out_of_corpus",
]


class GoldenQuery(BaseModel):
    """A single labeled evaluation query."""

    query_id: str
    text: str
    paper_id: str
    category: QueryCategory
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_pages: list[int] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)
    note: str | None = None


class GoldenSet(BaseModel):
    """A set of GoldenQueries used as one eval input. Versioned for reproducibility."""

    name: str
    version: str
    queries: list[GoldenQuery]


class RetrievalMetrics(BaseModel):
    """Per-query retrieval metrics. Macro-averaged in aggregates."""

    ndcg_at_5: float
    recall_at_10: float
    mrr: float


class GenerationMetrics(BaseModel):
    """Per-query generation metrics. Optional fields populated only when an
    LLM judge is configured (else `None`)."""

    citation_rate: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None


class PerQueryResult(BaseModel):
    """One golden query's full evaluation outcome."""

    query_id: str
    category: str
    text: str
    retrieved_chunk_ids: list[str]
    retrieval: RetrievalMetrics
    generation: GenerationMetrics | None = None
    answer_text: str | None = None
    cited_chunk_ids: list[str] = Field(default_factory=list)
    latency_ms: int = Field(ge=0)
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)


class EvalRun(BaseModel):
    """The top-level record of one evaluation run. Serialisable to JSON for the dashboard."""

    run_id: str
    started_at: datetime
    finished_at: datetime
    golden_set_name: str
    golden_set_version: str
    config: dict[str, Any] = Field(default_factory=dict)
    per_query: list[PerQueryResult]
