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
