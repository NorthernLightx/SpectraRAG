"""Figures endpoint: catalogue of figure-kind chunks across the corpus.

Sits next to /query (search-by-question) to handle the *browse* class of
intent — "show me figures" / "what plots does this paper have" — which
retrieval was never designed for. Reads from the chunk index populated at
lifespan startup; no Qdrant round trip per request.
"""

from __future__ import annotations

import re
from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from src.api.deps import get_chunks
from src.types import Chunk

router = APIRouter()

# Same regex + threshold as src/ingestion/docling_parser.py — duplicated
# here so the API can derive a `role` for legacy chunks ingested before
# ADR 0022 landed (no `role` in their metadata). New ingests carry the
# field; this branch is the migration cushion.
_FIGURE_CAPTION_RE = re.compile(r"^\s*(figure|fig\.?)\s*\d", re.IGNORECASE)
_MIN_FIGURE_AREA_PT2 = 5000.0
_PLACEHOLDER_CAPTION_RE = re.compile(r"^\s*\[.+::p\d+::fig\d+\]\s*$")


def _derive_role(caption: str, bbox: list[float] | None) -> Literal["figure", "decoration", "unlabeled"]:
    """Mirror of `docling_parser._classify_figure_role` for legacy chunks.

    The caption seen here is the chunk's emitted text (VLM caption ?
    PDF caption ? `[figure_id]` placeholder), so a placeholder string
    counts as "no real caption" — strip it before the Fig-N test.
    """
    real_caption = "" if _PLACEHOLDER_CAPTION_RE.match(caption) else caption
    if real_caption and _FIGURE_CAPTION_RE.match(real_caption):
        return "figure"
    if bbox is None:
        return "decoration"
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if area < _MIN_FIGURE_AREA_PT2:
        return "decoration"
    return "unlabeled"


class FigureBrowseItem(BaseModel):
    """One row in the figure catalogue.

    `caption` carries whatever text the ingestion pipeline picked as the
    figure's primary label — VLM caption when available, else the extracted
    caption, else a placeholder of `[figure_id]`. `bbox` is the figure's
    location on the page in PDF points (1/72 inch), set when extraction
    captured one (ADR 0009); demos use it to highlight the figure on the
    page. `role` (ADR 0022) tags decorative picture-detections — affiliation
    logos, license badges, inline status icons — that the gallery hides by
    default.
    """

    chunk_id: str
    paper_id: str
    page_number: int = Field(ge=1)
    caption: str
    bbox: list[float] | None = None
    has_vlm_caption: bool = False
    page_image_url: str
    role: Literal["figure", "decoration", "unlabeled"] = "unlabeled"


def _to_browse_item(chunk: Chunk) -> FigureBrowseItem | None:
    """Convert a figure-kind chunk to its browse representation. Returns None
    when the chunk doesn't have the expected shape (defensive — keeps the
    endpoint resilient to mid-migration corpora)."""
    if not chunk.page_numbers:
        return None
    page = chunk.page_numbers[0]
    bbox_raw = chunk.metadata.get("bbox")
    bbox: list[float] | None = None
    if (
        isinstance(bbox_raw, list)
        and len(bbox_raw) == 4
        and all(isinstance(v, (int, float)) for v in bbox_raw)
    ):
        bbox = [float(v) for v in bbox_raw]
    role_raw = chunk.metadata.get("role")
    if role_raw in {"figure", "decoration", "unlabeled"}:
        role: Literal["figure", "decoration", "unlabeled"] = role_raw
    else:
        role = _derive_role(chunk.text, bbox)
    return FigureBrowseItem(
        chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        page_number=page,
        caption=chunk.text,
        bbox=bbox,
        has_vlm_caption=bool(chunk.metadata.get("has_vlm_caption", False)),
        page_image_url=f"/pages/{chunk.paper_id}/{chunk.paper_id}_p{page}.png",
        role=role,
    )


@router.get("/figures", response_model=list[FigureBrowseItem])
def list_figures(
    paper_id: str | None = Query(default=None, description="Restrict to one paper."),
    limit: int = Query(default=200, ge=1, le=1000),
    chunks: dict[str, Chunk] = Depends(get_chunks),
) -> list[FigureBrowseItem]:
    out: list[FigureBrowseItem] = []
    for chunk in chunks.values():
        if chunk.metadata.get("kind") != "figure":
            continue
        if paper_id and chunk.paper_id != paper_id:
            continue
        item = _to_browse_item(chunk)
        if item is None:
            continue
        out.append(item)
        if len(out) >= limit:
            break
    # Stable order: paper → page → chunk_id (so the same corpus produces the
    # same listing across requests).
    out.sort(key=lambda x: (x.paper_id, x.page_number, x.chunk_id))
    return out
