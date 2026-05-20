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
import re
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.observability.logging import get_logger, timed_event
from src.types import Bbox, Figure, Table
from src.types.documents import FigureRole

# ADR 0022 — figure role classification. Docling's layout model labels
# affiliation logos, license badges, inline status icons, and small
# decorative glyphs as ``picture`` alongside real publication figures.
# Dropping them at ingestion is too aggressive (a "what license is this
# paper under" query loses its answer), so we tag every picture with a
# role and let the gallery / retriever filter where appropriate.
#
# 5000 pt² is the measured cut from the 20-paper arXiv-2604 corpus:
# 42 chunks below 1000 pt² (icons / logos), 1 chunk at 1130 (still an
# icon), zero chunks in 3k-8k (clean valley), 32 chunks 5k-20k all real
# figures (smallest: an 8789-pt² SWAP-test diagram). Captioned
# "Figure N" / "Fig. N" pictures pass regardless of area to rescue the
# rare small-but-labelled real figure (e.g. the 906-pt² "Figure 3:
# Screenshots of the artifacts ...").
_MIN_FIGURE_AREA_PT2 = 5000.0
_FIGURE_CAPTION_RE = re.compile(r"^\s*(figure|fig\.?)\s*\d", re.IGNORECASE)


def _classify_figure_role(*, caption: str, bbox: Bbox | None) -> FigureRole:
    """Deterministic figure-vs-decoration classifier (ADR 0022).

    Priority: a paper-authored "Figure N" caption beats every size heuristic
    — if the document labels it, it's a figure regardless of how small the
    crop is. Otherwise, sub-threshold-area pictures are ``decoration``
    (logos, icons, decorative glyphs). Everything else is ``unlabeled`` —
    a real picture that the paper didn't caption (e.g. an inset diagram
    inside an equation block); we keep it indexed but the gallery hides
    it from the default view.
    """
    if caption and _FIGURE_CAPTION_RE.match(caption):
        return "figure"
    if bbox is None:
        return "decoration"
    area = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0)
    if area < _MIN_FIGURE_AREA_PT2:
        return "decoration"
    return "unlabeled"

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


def convert_with_docling(pdf_path: Path) -> Any:
    """Single Docling conversion. Shared between text-chunking (ADR 0021) and
    figure / table extraction (ADR 0020) so we only run the layout +
    OCR pipeline once per paper. Returns the raw `DoclingDocument`."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return _build_converter().convert(pdf_path).document


def page_heights(doc: Any) -> dict[int, float]:
    """`{page_no: height_in_pt}` for `_flip_bbox` callers (TOP-LEFT origin)."""
    out: dict[int, float] = {}
    pages = getattr(doc, "pages", {})
    items = pages.items() if hasattr(pages, "items") else enumerate(pages, start=1)
    for page_no, page in items:
        size = getattr(page, "size", None)
        if size is None:
            continue
        try:
            out[int(page_no)] = float(size.height)
        except (AttributeError, TypeError, ValueError):
            continue
    return out


def parse_with_docling(
    paper_id: str,
    pdf_path: Path,
    *,
    out_dir: Path = Path("data/figures"),
    doc: Any | None = None,
) -> tuple[list[Figure], list[Table]]:
    """Run Docling over `pdf_path` and return Figures + Tables in project types.

    Per-item failures (image save error, bbox flip degenerate, markdown
    export error) log + skip rather than abort the whole conversion —
    same posture as `figures.py` / `captioner.py`.
    """
    if doc is None and not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with timed_event(_log, "docling.parsed", paper_id=paper_id, pdf=str(pdf_path)) as ctx:
        if doc is None:
            doc = convert_with_docling(pdf_path)
        heights = page_heights(doc)

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
            page_h = heights.get(page_no, 792.0)
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
            role = _classify_figure_role(caption=caption, bbox=bbox)
            figures.append(
                Figure(
                    figure_id=figure_id,
                    paper_id=paper_id,
                    page_number=page_no,
                    caption=caption,
                    image_path=image_path,
                    bbox=bbox,
                    role=role,
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
            page_h = heights.get(page_no, 792.0)
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
