"""Docling-based figure / table extraction (ADR 0020 primary fast path).

Replaces `extract_figures` (PyMuPDF embedded-XREF iteration) and
`extract_tables` (PyMuPDF `find_tables()` heuristic) with Docling's
deterministic layout + table-structure pipeline. Probe on `2604.22753v1`
recovered 7 / 7 expected figures and tables that the old extractors
silently missed (vector-only figures, tight-cell tables). Caption-to-
artifact linking is deterministic via Docling's document tree, not
regex matching across page text.

Output is `(list[Figure], list[Table])` using the existing types so
nothing downstream changes. Coordinate flip from Docling's
`CoordOrigin.BOTTOMLEFT` to the project's `Bbox` (TOP-LEFT origin, ADR
0009). Picture images saved to `data/figures/<paper_id>/` matching the
PyMuPDF path's filename convention.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.observability.logging import get_logger, timed_event
from src.types import Bbox, Figure, Table

_log = get_logger(__name__)

# 2x scale ~ 144 DPI for picture-image rasterisation. Matches the
# fidelity floor `captioner.py` expects; the project's page renderer
# uses 150 DPI elsewhere, so this stays in the same ballpark.
_PICTURE_SCALE = 2.0


def _safe_filename(figure_id: str) -> str:
    """Windows-safe filename: ``:`` is not a legal path character there."""
    return figure_id.replace(":", "_")


def _flip_bbox(raw: Any, page_height: float) -> Bbox | None:
    """Docling ``BoundingBox`` (``BOTTOMLEFT`` origin) → project ``Bbox`` (``TOP-LEFT``).

    Docling reports ``t`` (top) and ``b`` (bottom) in PDF-native
    BOTTOMLEFT coords — y=0 is the page bottom, so ``t > b`` and the
    visual top of the box has the *larger* y. Flipping to TOP-LEFT
    where y=0 is the page top: ``new_y_top = page_height - old_t`` and
    ``new_y_bottom = page_height - old_b``. The project's ``Bbox``
    validator requires ``y1 > y0`` and non-negative coords, so this
    only emits a value when the flip stays valid.
    """
    try:
        x0 = float(raw.l)
        x1 = float(raw.r)
        y0 = float(page_height - raw.t)
        y1 = float(page_height - raw.b)
    except (AttributeError, TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0 or x0 < 0 or y0 < 0:
        return None
    try:
        return Bbox(x0=x0, y0=y0, x1=x1, y1=y1)
    except ValueError:
        return None


def _build_converter() -> DocumentConverter:
    """Pdf converter with picture-image generation enabled so we can
    persist crops to disk (the project's `Figure.image_path` is required)."""
    pipeline = PdfPipelineOptions()
    pipeline.images_scale = _PICTURE_SCALE
    pipeline.generate_picture_images = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
    )


def parse_with_docling(
    paper_id: str,
    pdf_path: Path,
    *,
    out_dir: Path = Path("data/figures"),
) -> tuple[list[Figure], list[Table]]:
    """Run Docling over `pdf_path` and return Figures + Tables in project types.

    Per-item failures (image save error, bbox flip degenerate, markdown
    export error) log + skip rather than abort the whole conversion —
    same posture as `figures.py` / `captioner.py`.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with timed_event(_log, "docling.parsed", paper_id=paper_id, pdf=str(pdf_path)) as ctx:
        converter = _build_converter()
        result = converter.convert(pdf_path)
        doc = result.document

        # Page heights in PDF points, used to flip Docling's BOTTOMLEFT y
        # back to the project's TOP-LEFT Bbox.
        page_heights: dict[int, float] = {}
        pages = getattr(doc, "pages", {})
        items = pages.items() if hasattr(pages, "items") else enumerate(pages, start=1)
        for page_no, page in items:
            size = getattr(page, "size", None)
            if size is None:
                continue
            try:
                page_heights[int(page_no)] = float(size.height)
            except (AttributeError, TypeError, ValueError):
                continue

        paper_out = out_dir / paper_id
        paper_out.mkdir(parents=True, exist_ok=True)

        figures: list[Figure] = []
        for idx, pic in enumerate(getattr(doc, "pictures", []), start=1):
            provs = getattr(pic, "prov", None) or []
            if not provs:
                continue
            prov = provs[0]
            try:
                page_no = int(getattr(prov, "page_no", 0))
            except (TypeError, ValueError):
                page_no = 0
            if page_no <= 0:
                continue
            page_h = page_heights.get(page_no, 792.0)
            bbox = _flip_bbox(getattr(prov, "bbox", None), page_h)
            figure_id = f"{paper_id}::p{page_no}::fig{idx}"
            image_path = paper_out / f"{_safe_filename(figure_id)}.png"
            try:
                img = pic.get_image(doc)
            except (RuntimeError, AttributeError, KeyError) as exc:
                _log.warning("docling.figure_image_failed", figure_id=figure_id, error=str(exc))
                continue
            if img is None:
                continue
            try:
                img.save(image_path)
            except (OSError, ValueError) as exc:
                _log.warning("docling.figure_save_failed", figure_id=figure_id, error=str(exc))
                continue
            caption = ""
            with contextlib.suppress(RuntimeError, AttributeError):
                caption = str(pic.caption_text(doc) or "")
            figures.append(
                Figure(
                    figure_id=figure_id,
                    paper_id=paper_id,
                    page_number=page_no,
                    caption=caption,
                    image_path=image_path,
                    bbox=bbox,
                )
            )

        tables: list[Table] = []
        for idx, tab in enumerate(getattr(doc, "tables", []), start=1):
            provs = getattr(tab, "prov", None) or []
            if not provs:
                continue
            prov = provs[0]
            try:
                page_no = int(getattr(prov, "page_no", 0))
            except (TypeError, ValueError):
                page_no = 0
            if page_no <= 0:
                continue
            page_h = page_heights.get(page_no, 792.0)
            bbox = _flip_bbox(getattr(prov, "bbox", None), page_h)
            markdown = ""
            try:
                markdown = str(tab.export_to_markdown(doc) or "")
            except (RuntimeError, AttributeError) as exc:
                _log.debug("docling.table_markdown_failed", idx=idx, error=str(exc))
            if not markdown.strip():
                continue
            caption = ""
            with contextlib.suppress(RuntimeError, AttributeError):
                caption = str(tab.caption_text(doc) or "")
            tables.append(
                Table(
                    table_id=f"{paper_id}::p{page_no}::tab{idx}",
                    paper_id=paper_id,
                    page_number=page_no,
                    markdown=markdown,
                    caption=caption or None,
                    bbox=bbox,
                )
            )

        ctx["figures"] = len(figures)
        ctx["tables"] = len(tables)
        return figures, tables
