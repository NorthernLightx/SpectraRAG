"""Document-level Pydantic models shared across ingestion and retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Paper(BaseModel):
    """A scientific paper ingested into the corpus."""

    model_config = ConfigDict(frozen=True)

    paper_id: str
    arxiv_id: str | None = None
    title: str
    abstract: str | None = None
    pdf_path: Path
    authors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Page(BaseModel):
    """A rendered page of a paper. `image_path` populated only for the visual path."""

    paper_id: str
    page_number: int = Field(ge=1)
    text: str
    image_path: Path | None = None


class Chunk(BaseModel):
    """A retrievable text chunk produced by the ingestion pipeline."""

    chunk_id: str
    paper_id: str
    page_numbers: list[int] = Field(min_length=1)
    text: str
    section: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Figure(BaseModel):
    """A figure extracted from a paper. `vlm_caption` is set after VLM captioning."""

    figure_id: str
    paper_id: str
    page_number: int = Field(ge=1)
    caption: str
    image_path: Path
    vlm_caption: str | None = None


class Table(BaseModel):
    """A table extracted from a paper, normalised to markdown."""

    table_id: str
    paper_id: str
    page_number: int = Field(ge=1)
    markdown: str
    caption: str | None = None
