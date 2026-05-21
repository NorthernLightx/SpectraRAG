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
import os
import re
import shutil
from pathlib import Path
from typing import Any

# ADR 0022: Docling's DocumentFigureClassifier runs through transformers
# + torch.compile by default, which needs Triton (unavailable on Windows
# CPU). Disabling dynamo BEFORE importing docling sidesteps the compile
# path entirely. Set as early as possible so it propagates to torch on
# first import.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.observability.logging import get_logger, timed_event
from src.types import Bbox, Figure, Table
from src.types.documents import FigureRole

# ADR 0022 — figure role classification.
#
# Pictures get a coarse role (``figure``, ``decoration``, ``unlabeled``)
# so the gallery can hide page-furniture by default and retrieval can
# treat the buckets differently if needed. Two layers feed the role:
#
# 1. **Docling's DocumentFigureClassifier-v2.5** (preferred when enabled).
#    Outputs one of 28 fine-grained labels — `logo`, `icon`, `bar_chart`,
#    `flow_chart`, `photograph`, etc. — at high confidence on this
#    corpus (1.00 on every Microsoft-logo affiliation block, 0.97+ on
#    most figures). The label is preserved on chunk.metadata for richer
#    filtering; the role is derived from a fixed mapping below.
#
# 2. **Deterministic caption + area fallback** (kept for legacy
#    collections ingested before the classifier was wired in, and as a
#    safety net when the classifier returns nothing). 5000 pt² is the
#    measured cut from the 20-paper arXiv-2604 corpus: 42 chunks below
#    1000 pt² are icons/logos, 32 chunks 5k-20k are all real figures
#    (smallest a 8789-pt² SWAP-test diagram). Captioned "Figure N" /
#    "Fig. N" pictures pass regardless of area to rescue the rare
#    small-but-labelled real figure.
_MIN_FIGURE_AREA_PT2 = 5000.0
_FIGURE_CAPTION_RE = re.compile(
    r"""^\s*
        (?: \d+\s+ )?                       # optional leading page-number from OCR
        (?:
            (?:figure|fig\.?) \s+ [A-Z0-9]  # Figure 3, Fig. 3, Figure C.1, Figure F.
          | \([a-z]\)\s                     # (a), (b), ... subfigure caption
        )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Docling-label → our 3-role taxonomy. Every label the classifier knows
# about gets a deliberate mapping; anything new from a future model
# version falls into ``unlabeled``.
_DOCLING_LABEL_TO_ROLE: dict[str, FigureRole] = {
    # decorative / page-furniture
    "logo": "decoration",
    "icon": "decoration",
    "signature": "decoration",
    "stamp": "decoration",
    "bar_code": "decoration",
    "qr_code": "decoration",
    "page_thumbnail": "decoration",
    # real publication figures
    "bar_chart": "figure",
    "box_plot": "figure",
    "flow_chart": "figure",
    "line_chart": "figure",
    "pie_chart": "figure",
    "scatter_plot": "figure",
    "scatter_chart": "figure",
    "stacked_bar_chart": "figure",
    "heatmap": "figure",
    "photograph": "figure",
    "natural_image": "figure",
    "full_page_image": "figure",
    "screenshot_from_computer": "figure",
    "screenshot_from_manual": "figure",
    "screenshot": "figure",
    "chemistry_structure": "figure",
    "chemistry_molecular_structure": "figure",
    "chemistry_markush_structure": "figure",
    "engineering_drawing": "figure",
    "cad_drawing": "figure",
    "electrical_diagram": "figure",
    "geographical_map": "figure",
    "geographic_map": "figure",
    "map": "figure",
    "topographical_map": "figure",
    "remote_sensing": "figure",
    "stratigraphic_chart": "figure",
    "music": "figure",
    "picture_group": "figure",
    # Docling already extracts tables via a separate model; the picture-
    # detector firing on a table region is a duplicate, so we don't
    # claim it as a figure here.
    "table": "unlabeled",
    # explicitly uncertain
    "other": "unlabeled",
    "calendar": "unlabeled",
    "crossword_puzzle": "unlabeled",
}
# Below this confidence we don't trust the model — fall back to the
# caption/area heuristic. 0.30 is well above the uniform-prior baseline
# (~0.04 with 28 classes) but low enough to keep the model's good
# medium-confidence calls.
_MIN_CLASSIFIER_CONFIDENCE = 0.30


def _top_docling_label(pic: Any) -> tuple[str | None, float]:
    """Extract the top (class_name, confidence) from a Docling picture's
    classifier output. Returns (None, 0.0) when no classification is
    present. Reads ``pic.meta.classification`` (current API) with a
    fallback to ``pic.annotations`` (deprecated but still populated)
    so both Docling versions work."""
    # New-style: PictureMeta.classification.predictions
    meta = getattr(pic, "meta", None)
    classification = getattr(meta, "classification", None) if meta is not None else None
    preds = getattr(classification, "predictions", None) or []
    if not preds:
        # Old-style: list of PictureClassificationData with predicted_classes
        for ann in getattr(pic, "annotations", []) or []:
            preds = getattr(ann, "predicted_classes", None) or []
            if preds:
                break
    if not preds:
        return None, 0.0
    top = preds[0]
    return getattr(top, "class_name", None), float(getattr(top, "confidence", 0.0))


def _classify_figure_role(
    *, caption: str, bbox: Bbox | None, docling_label: str | None = None, confidence: float = 0.0
) -> FigureRole:
    """Pick a role for the picture (ADR 0022).

    Priority order — most authoritative signal first:

    1. **Paper-authored ``Figure N`` caption.** The paper telling us "this
       is Figure 3" beats the classifier. Catches the small-but-real
       Figure-3 / Figure-3-screenshot cases that the visual model can
       mistake for a logo because the thumbnail is so small.
    2. **Docling classifier label** at ≥ confidence threshold. 28 fine-
       grained labels — `logo`, `icon`, `bar_chart`, `flow_chart`, ... —
       mapped to our 3-role taxonomy via ``_DOCLING_LABEL_TO_ROLE``.
    3. **Area heuristic.** Sub-threshold pictures with neither a Figure-N
       caption nor a confident classifier label become ``decoration``;
       everything else is ``unlabeled``.
    """
    if caption and _FIGURE_CAPTION_RE.match(caption):
        return "figure"
    if (
        docling_label is not None
        and confidence >= _MIN_CLASSIFIER_CONFIDENCE
        and docling_label in _DOCLING_LABEL_TO_ROLE
    ):
        return _DOCLING_LABEL_TO_ROLE[docling_label]
    if bbox is None:
        # No bbox to place or measure → "unknown", not "page furniture". ADR
        # 0022: `decoration` is removed from the gallery content view, excluded
        # by the role-aware retrieval filter, and skipped by the VLM captioner —
        # so defaulting a bbox-less *real* figure to decoration would silently
        # delete it end to end. `unlabeled` keeps it retrievable + captionable.
        return "unlabeled"
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
    persist crops to disk (the project's `Figure.image_path` is required).
    Picture classification is on (ADR 0022) — produces the role label;
    `TORCHDYNAMO_DISABLE=1` is set at import time to keep the underlying
    transformers engine off the torch.compile path."""
    pipeline = PdfPipelineOptions()
    pipeline.images_scale = _PICTURE_SCALE
    pipeline.generate_picture_images = True
    pipeline.do_picture_classification = True
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
        # Clear any prior run's crops first. Figure ids are per-run (global
        # picture index), so re-ingesting — especially with a different
        # extractor — otherwise leaves orphaned crops that collide with new ids
        # and turn the dir into a palimpsest that misleads anything reading it.
        if paper_out.exists():
            shutil.rmtree(paper_out, ignore_errors=True)
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
            docling_label, confidence = _top_docling_label(pic)
            role = _classify_figure_role(
                caption=caption,
                bbox=bbox,
                docling_label=docling_label,
                confidence=confidence,
            )
            figures.append(
                Figure(
                    figure_id=figure_id,
                    paper_id=paper_id,
                    page_number=page_no,
                    caption=caption,
                    image_path=image_path,
                    bbox=bbox,
                    role=role,
                    docling_label=docling_label,
                    docling_label_confidence=confidence,
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
