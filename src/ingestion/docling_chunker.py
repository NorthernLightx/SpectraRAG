"""Document-level text chunking from a `DoclingDocument` (ADR 0021).

Replaces the PyMuPDF + ADR-0017 regex-based chunker for the text path
when `use_docling=True`. Three structural wins over the prior path
(probe in `scripts/experiments/docling_text_probe.py`):

1. **Reading order is layout-aware.** Multi-column papers no longer
   glue column-1 + column-2 into one read; Docling's layout model
   orders blocks correctly.
2. **Section boundaries are deterministic.** Numbered headers PyMuPDF
   used to merge into body text (defeating the ADR-0017 regex) are
   labelled `section_header` by Docling. No more "Abstract"-everything
   misattribution.
3. **Page furniture + figure-interior text are filtered by label, not
   guessed.** `page_header` / `page_footer` blocks are dropped
   deterministically; body text whose bbox falls inside a figure or
   table region is excluded (no more leaked axis-tick / value-grid
   strings polluting body chunks).

Per-chunk `metadata['bbox']` carries the union of contributing-block
bboxes for single-page chunks, enabling region-precise citations on
text answers (ADR 0009 extends from figures/tables to text).
"""

from __future__ import annotations

from typing import Any, NamedTuple

from src.ingestion.chunking import _window_spans
from src.ingestion.docling_parser import _flip_bbox, page_heights
from src.observability.logging import get_logger, timed_event
from src.types import Bbox, Chunk

_log = get_logger(__name__)

# Labels Docling assigns that we *include* in the chunked text body.
# Anything else (page_header, page_footer, picture, table, caption,
# title) is either page furniture or owned by a separate chunk type.
_BODY_LABELS = frozenset({"text", "list_item", "formula", "footnote", "code", "paragraph"})

# Minimum block char length to include — drops one-character axis-tick
# leakage that wasn't fully caught by the figure-bbox containment check.
_MIN_BLOCK_CHARS = 2

# Threshold for "this text block is inside this figure/table" containment.
_INSIDE_FRACTION = 0.8


class _Block(NamedTuple):
    """One normalised text block en route to a Chunk."""

    text: str
    label: str
    page: int
    bbox: Bbox | None


def _bbox_inside_any(inner: Bbox, outers: list[Bbox]) -> bool:
    """True when ``inner`` is mostly (>=80% of its area) contained by one
    of ``outers``. Used to drop text blocks that Docling identifies as
    sitting *inside* a figure / table region — those are figure-interior
    leak text we don't want in body chunks."""
    inner_area = (inner.x1 - inner.x0) * (inner.y1 - inner.y0)
    if inner_area <= 0:
        return False
    for outer in outers:
        if inner.x1 <= outer.x0 or inner.x0 >= outer.x1:
            continue
        if inner.y1 <= outer.y0 or inner.y0 >= outer.y1:
            continue
        ix0 = max(inner.x0, outer.x0)
        iy0 = max(inner.y0, outer.y0)
        ix1 = min(inner.x1, outer.x1)
        iy1 = min(inner.y1, outer.y1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        if inter / inner_area >= _INSIDE_FRACTION:
            return True
    return False


def _figure_table_bboxes_by_page(doc: Any) -> dict[int, list[Bbox]]:
    """Collect figure + table bboxes per page (TOP-LEFT) so the body
    chunker can drop any text block whose bbox falls inside them."""
    heights = page_heights(doc)
    by_page: dict[int, list[Bbox]] = {}
    for kind in ("pictures", "tables"):
        for item in getattr(doc, kind, []):
            provs = getattr(item, "prov", None) or []
            if not provs:
                continue
            prov = provs[0]
            try:
                page_no = int(getattr(prov, "page_no", 0))
            except (TypeError, ValueError):
                continue
            if page_no <= 0:
                continue
            page_h = heights.get(page_no, 792.0)
            bbox = _flip_bbox(getattr(prov, "bbox", None), page_h)
            if bbox is not None:
                by_page.setdefault(page_no, []).append(bbox)
    return by_page


def _union_bbox(bboxes: list[Bbox]) -> Bbox | None:
    if not bboxes:
        return None
    x0 = min(b.x0 for b in bboxes)
    y0 = min(b.y0 for b in bboxes)
    x1 = max(b.x1 for b in bboxes)
    y1 = max(b.y1 for b in bboxes)
    try:
        return Bbox(x0=x0, y0=y0, x1=x1, y1=y1)
    except ValueError:
        return None


def _accept_block(
    item: Any,
    *,
    heights: dict[int, float],
    fig_tab_by_page: dict[int, list[Bbox]],
) -> _Block | None:
    """Normalize one Docling text item into a `_Block` if it's body-eligible.

    Returns None when the item is page furniture, caption / title (owned
    by other chunk types), a figure-interior text leak, or too short to
    be meaningful prose.
    """
    label = str(getattr(item, "label", "")).strip().lower()
    if label not in _BODY_LABELS:
        return None
    text = (getattr(item, "text", "") or "").strip()
    if len(text) < _MIN_BLOCK_CHARS:
        return None
    provs = getattr(item, "prov", None) or []
    if not provs:
        return None
    prov = provs[0]
    try:
        page_no = int(getattr(prov, "page_no", 0))
    except (TypeError, ValueError):
        return None
    if page_no <= 0:
        return None
    page_h = heights.get(page_no, 792.0)
    bbox = _flip_bbox(getattr(prov, "bbox", None), page_h)
    if bbox is not None and _bbox_inside_any(bbox, fig_tab_by_page.get(page_no, [])):
        return None
    return _Block(text=text, label=label, page=page_no, bbox=bbox)


def chunk_with_docling(
    paper_id: str,
    doc: Any,
    *,
    target_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[Chunk]:
    """Walk `doc.texts` in reading order, accumulate body blocks under their
    `section_header`, window each section into Chunks with proper page +
    bbox metadata.
    """
    heights = page_heights(doc)
    fig_tab = _figure_table_bboxes_by_page(doc)

    # Single pass: accumulate body blocks under the current section_header.
    sections: list[tuple[str | None, list[_Block]]] = []
    current_title: str | None = None
    current_body: list[_Block] = []
    for item in getattr(doc, "texts", []):
        label = str(getattr(item, "label", "")).strip().lower()
        if label == "section_header":
            if current_body:
                sections.append((current_title, current_body))
                current_body = []
            new_title = (getattr(item, "text", "") or "").strip()
            current_title = new_title or current_title
            continue
        block = _accept_block(item, heights=heights, fig_tab_by_page=fig_tab)
        if block is not None:
            current_body.append(block)
    if current_body:
        sections.append((current_title, current_body))

    chunks: list[Chunk] = []
    counter = 0
    with timed_event(
        _log,
        "docling_chunk.done",
        paper_id=paper_id,
        sections=len(sections),
        total_blocks=sum(len(body) for _, body in sections),
    ) as ctx:
        for title, body in sections:
            # One joined string per section, then window it. Block boundary
            # is "\n" so the offsets line up for the bbox-contribution scan.
            body_text = "\n".join(b.text for b in body)
            if not body_text.strip():
                continue
            for win_start, win_end in _window_spans(body_text, target_chars, overlap_chars):
                window_text = body_text[win_start:win_end].strip()
                if not window_text:
                    continue
                # Walk blocks in order, tracking char offsets, to find which
                # blocks contribute to this window. Same "\n" separator.
                cursor = 0
                pages: set[int] = set()
                contrib_bboxes: list[Bbox] = []
                for b in body:
                    block_end = cursor + len(b.text)
                    if cursor < win_end and block_end > win_start:
                        pages.add(b.page)
                        if b.bbox is not None and b.page == b.page:
                            contrib_bboxes.append(b.bbox)
                    cursor = block_end + 1  # +1 for the join '\n'
                if not pages:
                    continue
                page_numbers = sorted(pages)
                metadata: dict[str, Any] = {}
                # bbox: emit the union of contributing-block bboxes only when
                # the chunk is single-page; cross-page chunks have no single
                # rectangle (ADR 0009's Bbox is page-local).
                if len(page_numbers) == 1:
                    union = _union_bbox(contrib_bboxes)
                    if union is not None:
                        metadata["bbox"] = union.as_list()
                chunks.append(
                    Chunk(
                        chunk_id=f"{paper_id}::p{page_numbers[0]}::c{counter}",
                        paper_id=paper_id,
                        page_numbers=page_numbers,
                        text=window_text,
                        section=title,
                        metadata=metadata,
                    )
                )
                counter += 1
        ctx["chunks"] = len(chunks)
    return chunks


def paper_text_from_docling(doc: Any) -> str:
    """Concatenated body text (no page furniture, no figure-interior),
    suitable as the long-form input to `contextualize_chunks`. Mirrors
    what `"\\n\\n".join(p.text for p in pages)` produced for the
    PyMuPDF path but cleaner."""
    heights = page_heights(doc)
    fig_tab = _figure_table_bboxes_by_page(doc)
    parts: list[str] = []
    for item in getattr(doc, "texts", []):
        block = _accept_block(item, heights=heights, fig_tab_by_page=fig_tab)
        if block is not None:
            parts.append(block.text)
    return "\n\n".join(parts)
