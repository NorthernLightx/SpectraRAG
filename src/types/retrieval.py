"""Retrieval-stage Pydantic models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.types.documents import Chunk

RetrievalSource = Literal["pipeline", "visual"]


class Query(BaseModel):
    """A user query against the RAG system."""

    text: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    filters: dict[str, Any] = Field(default_factory=dict)
    # ADR 0008: optional override that bypasses the routing classifier.
    # Used by the eval harness ("run every query through hybrid for the
    # comparison") and for debugging. Not exposed in the public OpenAPI
    # schema (caveat §5 in ADR 0008).
    force_route: Literal["text", "hybrid"] | None = None
    # ADR 0010: per-query routing-mode override. When None, the server's
    # configured mode is used. When set, RoutingRetriever switches dispatch
    # logic for this call only. Cascade dispatch falls back to a 0.85
    # threshold when the server wasn't started with one configured.
    routing_mode: Literal["category", "cascade"] | None = None

    def paper_id_filter(self) -> str | None:
        """Optional single-paper scope hint (ADR 0009 follow-up). Eval populates
        ``filters['paper_id']`` from GoldenQuery.paper_id; production callers
        pass nothing. Returns the id only when it's a string, else None."""
        value = self.filters.get("paper_id")
        return value if isinstance(value, str) else None


class RetrievalResult(BaseModel):
    """A single retrieved item with provenance, before reranking."""

    chunk_id: str
    paper_id: str
    score: float
    text: str
    page_numbers: list[int] = Field(min_length=1)
    source: RetrievalSource
    metadata: dict[str, Any] = Field(default_factory=dict)


class RankedChunk(BaseModel):
    """A retrieved chunk with rank and optional rerank score."""

    chunk: Chunk
    score: float
    rerank_score: float | None = None
    rank: int = Field(ge=1)


class RoutingInfo(BaseModel):
    """Per-call routing decision surfaced to the API caller.

    Captures what RoutingRetriever did for this query — which mode it ran in,
    which path it chose, and (for cascade) the confidence-based decision.
    The /query endpoint includes this so the demo UI can show "routed: ..."
    next to timings.
    """

    mode: Literal["category", "cascade"]
    path: Literal["text", "hybrid"]
    forced: bool = False
    category: str | None = None  # set only when mode=category
    cascade_decision: str | None = None  # set only when mode=cascade
    cascade_top_score: float | None = None
    cascade_threshold: float | None = None
    visual_failed: bool = False


class RetrievalResponse(BaseModel):
    """Wrapper for /query — chunks plus the routing decision that produced them.

    `routing` is None when the wired retriever doesn't expose a decision (e.g.
    PipelineRetriever with no routing layer).
    """

    results: list[RetrievalResult]
    routing: RoutingInfo | None = None
