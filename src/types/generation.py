"""Generation-stage Pydantic models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.types.retrieval import RankedChunk, RetrievalResult


class Citation(BaseModel):
    """A citation pointing back to a retrieved chunk."""

    chunk_id: str
    paper_id: str
    page_numbers: list[int] = Field(min_length=1)
    quote: str | None = None


class Context(BaseModel):
    """Assembled context handed to the generator."""

    query: str
    chunks: list[RankedChunk]
    token_count: int = Field(ge=0)


class Answer(BaseModel):
    """A generated answer with citations, model identity, and cost/latency.

    `retrieved` is populated by the API route layer (not the Generator) so
    callers can render "what the LLM saw" alongside the answer — used by the
    bundled web UI to show retrieved chunks + their source ('pipeline' vs
    'visual'). Defaults to [] so eval / unit-test code paths that construct
    Answer directly stay backwards-compatible.
    """

    text: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[RetrievalResult] = Field(default_factory=list)
    model: str
    prompt_version: str | None = None
    latency_ms: int = Field(ge=0)
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
