"""Document-level Pydantic models shared across ingestion and retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Bbox(BaseModel):
    """A bounding box in PDF coordinate space (points, 1/72 inch).

    Origin is the page's top-left corner per PyMuPDF's convention (note: PDF's
    raw coordinate origin is bottom-left, but PyMuPDF's `Rect` exposes
    top-left so consumers don't have to flip). `x0,y0` is the top-left of
    the box; `x1,y1` is the bottom-right; both inclusive of the box edge.

    Conversion to rendered-PNG pixel coordinates: multiply by `dpi / 72`.
    See ADR 0009 for why bbox stays in PDF points at this layer.
    """

    model_config = ConfigDict(frozen=True)

    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)

    @model_validator(mode="after")
    def _check_ordering(self) -> Bbox:
        if self.x1 <= self.x0:
            raise ValueError(f"Bbox: x1 ({self.x1}) must be > x0 ({self.x0})")
        if self.y1 <= self.y0:
            raise ValueError(f"Bbox: y1 ({self.y1}) must be > y0 ({self.y0})")
        return self

    def as_list(self) -> list[float]:
        """Pack as `[x0, y0, x1, y1]` for storage in `Chunk.metadata['bbox']`."""
        return [self.x0, self.y0, self.x1, self.y1]


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
    """A retrievable text chunk produced by the ingestion pipeline.

    `context` holds an optional LLM-generated blurb (Anthropic-style contextual
    retrieval) prepended to `text` at index time so the chunk is embedded *with*
    its situating context. Display-time citations should still use `text`.
    """

    chunk_id: str
    paper_id: str
    page_numbers: list[int] = Field(min_length=1)
    text: str
    section: str | None = None
    context: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def indexed_text(self) -> str:
        """Text used for embedding + BM25 indexing — context-prepended if present."""
        if self.context:
            return f"{self.context}\n\n{self.text}"
        return self.text


class Figure(BaseModel):
    """A figure extracted from a paper. `vlm_caption` is set after VLM captioning.

    `bbox` is the figure's location on the page in PDF points; absent (None)
    when PyMuPDF can't locate the embedded image stream on the page (rare —
    happens with vector-art figures and transparent overlays). ADR 0009.
    """

    figure_id: str
    paper_id: str
    page_number: int = Field(ge=1)
    caption: str
    image_path: Path
    vlm_caption: str | None = None
    bbox: Bbox | None = None


class Table(BaseModel):
    """A table extracted from a paper, normalised to markdown.

    `bbox` is PyMuPDF's detected table rectangle in PDF points; ADR 0009.
    None when the detector returns no rect (heuristic detection failure).
    """

    table_id: str
    paper_id: str
    page_number: int = Field(ge=1)
    markdown: str
    caption: str | None = None
    bbox: Bbox | None = None
