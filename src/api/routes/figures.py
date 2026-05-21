"""Document-element gallery endpoint.

Sits next to /query (search-by-question) to handle the *browse* class of
intent — "show me figures", "what plots does this paper have", "list
the tables" — which retrieval was never designed for. Reads from the
chunk index populated at lifespan startup; no Qdrant round trip per
request.

The endpoint historically returned only `kind=figure` chunks (legacy
name `/figures`). It now also surfaces `kind=table` chunks and the
fine-grained `docling_label` (ADR 0022) so the gallery can offer
sub-label selectors (`bar_chart`, `flow_chart`, `logo`, `table`, …)
instead of the old 2-button "figures only / all detections" toggle.
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
_FIGURE_CAPTION_RE = re.compile(
    r"""^\s*
        (?: \d+\s+ )?
        (?:
            (?:figure|fig\.?) \s+ [A-Z0-9]
          | \([a-z]\)\s
        )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_MIN_FIGURE_AREA_PT2 = 5000.0
# Gallery floor (ADR 0022 amendment): decoration-role detections below this
# displayed area are glyph-sized page noise — 12-pt inline table emoji / icons
# (e.g. 2604.28177v1 p2 had 6 detections at 146 pt²). Hidden from the gallery
# even in the opt-in Decorative bucket. Decoration-only, so it can't drop a
# real figure: the smallest in-corpus figure is a 514-pt² caption-rescued one,
# and real figures are never role=decoration.
_MIN_DECORATION_AREA_PT2 = 500.0
_PLACEHOLDER_CAPTION_RE = re.compile(r"^\s*\[.+::p\d+::(?:fig|tab)\d+\]\s*$")

Role = Literal["figure", "decoration", "unlabeled"]
Kind = Literal["figure", "table"]


def _derive_role(caption: str, bbox: list[float] | None) -> Role:
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
    """One row in the gallery catalogue.

    `caption` carries whatever text the ingestion pipeline picked as the
    item's primary label — VLM caption when available, else the extracted
    caption, else a placeholder of `[chunk_id]`. `bbox` is the item's
    location on the page in PDF points (1/72 inch), set when extraction
    captured one (ADR 0009); demos use it to highlight the region.
    `kind` separates figures from tables; both share the gallery surface.
    `docling_label` (ADR 0022) is Docling's fine-grained classification
    for figure-kind chunks (`logo`, `bar_chart`, `flow_chart`, …); `None`
    for tables and for legacy figure chunks ingested before the
    classifier was wired in.
    """

    chunk_id: str
    paper_id: str
    page_number: int = Field(ge=1)
    caption: str
    bbox: list[float] | None = None
    has_vlm_caption: bool = False
    page_image_url: str
    kind: Kind = "figure"
    role: Role = "unlabeled"
    docling_label: str | None = None
    docling_label_confidence: float | None = None


def _to_browse_item(chunk: Chunk) -> FigureBrowseItem | None:
    """Convert a figure- or table-kind chunk to its browse representation.

    Returns ``None`` when the chunk doesn't have the expected shape
    (defensive — keeps the endpoint resilient to mid-migration corpora).
    """
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

    kind_raw = chunk.metadata.get("kind")
    kind: Kind = "table" if kind_raw == "table" else "figure"

    # Tables are always "figure"-role for gallery purposes (they're real
    # content, never decoration). Figures use the stored role or fall
    # back to the caption + area heuristic.
    if kind == "table":
        role: Role = "figure"
    else:
        role_raw = chunk.metadata.get("role")
        if role_raw in {"figure", "decoration", "unlabeled"}:
            role = role_raw
        else:
            role = _derive_role(chunk.text, bbox)

    docling_label_raw = chunk.metadata.get("docling_label")
    docling_label = docling_label_raw if isinstance(docling_label_raw, str) else None
    conf_raw = chunk.metadata.get("docling_label_confidence")
    confidence = float(conf_raw) if isinstance(conf_raw, (int, float)) else None

    return FigureBrowseItem(
        chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        page_number=page,
        caption=chunk.text,
        bbox=bbox,
        has_vlm_caption=bool(chunk.metadata.get("has_vlm_caption", False)),
        page_image_url=f"/pages/{chunk.paper_id}/{chunk.paper_id}_p{page}.png",
        kind=kind,
        role=role,
        docling_label=docling_label,
        docling_label_confidence=confidence,
    )


def _is_tiny_decoration(item: FigureBrowseItem) -> bool:
    """True for glyph-sized decoration detections to hide from the gallery.

    Decoration-only by design (real figures are never ``role=decoration``), so
    the area floor cannot drop a legitimate figure. ``bbox``-less items are kept.
    """
    if item.role != "decoration" or item.bbox is None:
        return False
    area = (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
    return area < _MIN_DECORATION_AREA_PT2


@router.get("/figures", response_model=list[FigureBrowseItem])
def list_figures(
    paper_id: str | None = Query(default=None, description="Restrict to one paper."),
    limit: int = Query(default=200, ge=1, le=1000),
    chunks: dict[str, Chunk] = Depends(get_chunks),
) -> list[FigureBrowseItem]:
    out: list[FigureBrowseItem] = []
    for chunk in chunks.values():
        kind = chunk.metadata.get("kind")
        if kind not in {"figure", "table"}:
            continue
        if paper_id and chunk.paper_id != paper_id:
            continue
        item = _to_browse_item(chunk)
        if item is None:
            continue
        if _is_tiny_decoration(item):
            continue
        out.append(item)
        if len(out) >= limit:
            break
    # Stable order: paper → page → chunk_id (so the same corpus produces the
    # same listing across requests).
    out.sort(key=lambda x: (x.paper_id, x.page_number, x.chunk_id))
    return out
