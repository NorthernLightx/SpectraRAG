"""Figure extraction from PDFs.

PyMuPDF gives us two views of figures:
- Embedded image streams (via `page.get_images()`) — the bytes that the PDF
  embeds. Unique-by-XREF, so the same logo across pages is one figure not N.
- Rendered drawings (vector art) — not exported here; covered later if needed.

The logical retrieval unit is one `Figure N:` caption span, not one XREF.
Many PDFs encode a single composite figure (e.g., a 5x2 grid of class-panels,
each panel a 4x4 grid of model-output thumbnails) as 100+ separate XREFs.
Indexing each XREF as its own chunk produces noise; the information lives at
the aggregate level (one logical figure with one caption). ADR 0011 covers
this — extraction now anchors on captions and bundles co-located XREFs.

Caption association: we parse `Figure N:` / `Fig. N:` labels from page text
(with block-level bbox detection so we know *where* each caption sits), then
assign each XREF on the page to the caption whose label is nearest in
y-coordinate. The representative image for the logical figure is the
largest XREF in the group; the figure's bbox is the union of all member
XREF bboxes (so citations point at the whole composite, not one cell).
Pages without any captions fall back to per-XREF extraction.

Each Figure gets `caption` set to the PDF-extracted text. `vlm_caption` is
left None at this layer; a later pass can fill it from a vision model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz

from src.observability.logging import get_logger
from src.types import Bbox, Figure

_log = get_logger(__name__)

# Match `Figure 12:`, `Fig. 3 —`, `Figure E.1:`, `Figure S1:`, `Fig. 4.2:`.
# Appendix and supplementary figures use letter prefixes (`E.1`, `S1`, `A.3`)
# that the prior `\d+`-only pattern missed — paper 2604.28190v1 had ~900
# appendix figure XREFs whose `Figure E.1:` caption never bound to an anchor,
# so the aggregation pass fell through to per-XREF emission. The body runs
# until the next blank line (paragraph break) or another Figure/Table label,
# whichever comes first. `[\s\S]` is "any char including newlines" without
# flipping the whole pattern into DOTALL mode.
_FIG_NUMBER_PATTERN = r"(?:[A-Z]\.?)?\d+(?:\.\d+)*"
_FIG_LABEL_RE = re.compile(
    rf"(?:Figure|Fig\.?)\s+({_FIG_NUMBER_PATTERN})\s*[:.\-—]\s*([\s\S]*?)"
    rf"(?=\n\s*\n|\n\s*(?:Figure|Fig\.?|Table)\s+{_FIG_NUMBER_PATTERN}\s*[:.\-—]|\Z)",
    re.IGNORECASE,
)

# Lighter-weight regex used only to detect *which block* a caption label lives
# in (for bbox attachment). The label itself anchors at the start of the match;
# we don't care about the body here — that comes from `_extract_captions`.
_FIG_LABEL_ANCHOR_RE = re.compile(
    rf"(?:Figure|Fig\.?)\s+({_FIG_NUMBER_PATTERN})\s*[:.\-—]", re.IGNORECASE
)


@dataclass(frozen=True)
class _XrefRecord:
    """One embedded image stream on a page (post-min_dim filter)."""

    xref: int
    bbox: Bbox | None
    width: int
    height: int


def _safe_index(paper_id: str, page_no: int, idx: int) -> str:
    return f"{paper_id}::p{page_no}::fig{idx}"


def _safe_filename(figure_id: str) -> str:
    """Convert `paper::p1::fig1` into a Windows-safe filename — `:` is not legal there."""
    return figure_id.replace(":", "_")


def _extract_captions(page_text: str) -> dict[str, str]:
    """Return `{figure_label: caption_body}` parsed from page text.

    `figure_label` is the raw identifier as written in the PDF — `"1"`,
    `"12"`, `"E.1"`, `"S1"`, etc. Keys are strings (not ints) because
    appendix and supplementary figures use letter-prefixed labels.
    """
    captions: dict[str, str] = {}
    for match in _FIG_LABEL_RE.finditer(page_text):
        label = match.group(1).strip()
        if not label:
            continue
        body = " ".join(match.group(2).split())
        # Truncate at the first paragraph break (blank line) — long captions
        # are often followed by body text we don't want to swallow.
        if "\n\n" in body:
            body = body.split("\n\n", 1)[0].strip()
        captions[label] = f"Figure {label}: {body}".strip()
    return captions


def _captions_with_bboxes(page: fitz.Page) -> dict[str, tuple[str, Bbox | None]]:
    """Return `{label: (caption_text, label_bbox)}` for `Figure N:` labels on the page.

    The text body comes from `_extract_captions` (full-text regex, so multi-line
    bodies survive). The bbox is taken from the PyMuPDF text *block* that
    contains the `Figure N:` label — that block's rectangle is what we'll use
    to assign nearby XREFs in `_assign_xrefs_to_captions`. Multi-block captions
    keep only the label block's bbox; that's fine because we only need it as a
    spatial anchor for nearest-neighbour assignment.

    If `get_text("blocks")` doesn't expose a block matching the label (rare —
    PyMuPDF returns malformed block tuples on some encrypted PDFs), the
    caption's bbox is None and its XREFs fall through to the per-XREF
    fallback path in `extract_figures`.
    """
    page_text = page.get_text("text") or ""
    captions = _extract_captions(page_text)
    if not captions:
        return {}

    bbox_by_label: dict[str, Bbox | None] = dict.fromkeys(captions)
    try:
        blocks = page.get_text("blocks") or []
    except (RuntimeError, ValueError) as exc:
        _log.debug("figure.blocks_unavailable", error=str(exc))
        return {label: (captions[label], None) for label in captions}

    for block in blocks:
        # PyMuPDF block tuple: (x0, y0, x1, y1, text, block_no, block_type, ...).
        if len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
        if not isinstance(text, str):
            continue
        for match in _FIG_LABEL_ANCHOR_RE.finditer(text):
            label = match.group(1).strip()
            if label not in captions or bbox_by_label[label] is not None:
                continue
            try:
                bbox_by_label[label] = Bbox(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1))
            except (ValueError, TypeError) as exc:
                _log.debug("figure.caption_bbox_invalid", label=label, error=str(exc))

    return {label: (captions[label], bbox_by_label[label]) for label in captions}


def _save_pixmap(doc: fitz.Document, xref: int, out_path: Path) -> None:
    """Render an XREF as a PNG, converting to RGB if needed (alpha + CMYK both fail save)."""
    pix = fitz.Pixmap(doc, xref)
    if pix.alpha or pix.colorspace is None or pix.colorspace.name != "DeviceRGB":
        pix = fitz.Pixmap(fitz.csRGB, pix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))


def _figure_bbox(page: fitz.Page, xref: int) -> Bbox | None:
    """Return the bbox where `xref` is placed on `page`, or None.

    PyMuPDF's `get_image_rects(xref)` returns the page-local rects where the
    image is referenced (one xref can be placed multiple times on a single
    page — e.g. a logo header on a wide layout). We take the first rect; for
    figure-style images with one placement that's the right answer, and for
    multi-placed images the first is good enough for citation purposes.

    Returns None when the rect list is empty (vector-art "images" that
    PyMuPDF tracks in xref but doesn't place via Image XObject), when the
    rect has negative coordinates (off-page placement via CTM transformation
    on the source PDF), or when Bbox validation fails for any other reason.
    These failure modes are common on dense ArXiv figures; we log at debug,
    not warning, to keep ingestion logs readable. ADR 0009 §"Failure modes" #1.
    """
    try:
        rects = page.get_image_rects(xref)
    except (RuntimeError, ValueError, AttributeError) as exc:
        # AttributeError trips on very old fitz builds without get_image_rects.
        _log.debug("figure.bbox_unavailable", xref=xref, error=str(exc))
        return None
    if not rects:
        return None
    rect = rects[0]
    try:
        return Bbox(x0=float(rect.x0), y0=float(rect.y0), x1=float(rect.x1), y1=float(rect.y1))
    except (ValueError, TypeError) as exc:
        # Off-page CTM, degenerate rect (zero w/h), or non-numeric coords.
        _log.debug("figure.bbox_invalid", xref=xref, rect=str(rect), error=str(exc))
        return None


def _union_bbox(bboxes: list[Bbox]) -> Bbox | None:
    """Return the axis-aligned union of a non-empty bbox list, or None on degenerate input.

    Used to compute one Figure's bbox from N member XREFs that share a caption
    (composite figure). A composite's union rect is the natural citation
    target — clicking it highlights the whole grid, not one panel.
    """
    if not bboxes:
        return None
    x0 = min(b.x0 for b in bboxes)
    y0 = min(b.y0 for b in bboxes)
    x1 = max(b.x1 for b in bboxes)
    y1 = max(b.y1 for b in bboxes)
    try:
        return Bbox(x0=x0, y0=y0, x1=x1, y1=y1)
    except (ValueError, TypeError):
        return None


def _assign_xrefs_to_captions(
    xrefs: list[_XrefRecord],
    captions: dict[str, tuple[str, Bbox | None]],
) -> tuple[dict[str, list[_XrefRecord]], list[_XrefRecord]]:
    """Group XREFs under the nearest `Figure N:` caption by vertical distance.

    Heuristic: each XREF goes to the caption whose label bbox has the closest
    y-center. Captions on a page roughly partition it into vertical bands,
    so nearest-y is a good first approximation. Multi-XREF composite figures
    all cluster around one caption naturally; multi-figure pages partition
    XREFs across captions.

    Returns:
        `(by_label, unassigned)`:
        - `by_label[label]` = list of XREFs that map to that caption. May be empty.
        - `unassigned` = XREFs that couldn't be assigned (XREF has no bbox,
          or no caption on the page has a bbox to anchor against). These
          fall through to per-XREF extraction in `extract_figures`.
    """
    bboxed_captions = {label: bbox for label, (_text, bbox) in captions.items() if bbox is not None}
    if not bboxed_captions:
        return {}, list(xrefs)

    by_label: dict[str, list[_XrefRecord]] = {label: [] for label in bboxed_captions}
    unassigned: list[_XrefRecord] = []

    for record in xrefs:
        if record.bbox is None:
            unassigned.append(record)
            continue
        xref_center_y = (record.bbox.y0 + record.bbox.y1) / 2

        def _distance(label: str, _y: float = xref_center_y) -> float:
            caption_bbox = bboxed_captions[label]
            caption_center_y = (caption_bbox.y0 + caption_bbox.y1) / 2
            return abs(caption_center_y - _y)

        nearest = min(bboxed_captions, key=_distance)
        by_label[nearest].append(record)

    return by_label, unassigned


def extract_figures(
    paper_id: str,
    pdf_path: Path,
    *,
    out_dir: Path = Path("data/figures"),
    min_dim: int = 64,
) -> list[Figure]:
    """Extract logical figures from `pdf_path`; save PNGs under `out_dir/<paper_id>/`.

    Caption-anchored aggregation (ADR 0011): groups XREFs on each page under
    the nearest `Figure N:` caption and emits ONE `Figure` per caption with
    the largest member XREF as the representative `image_path` and the union
    of member bboxes as `bbox`. Pages without parseable captions fall back to
    per-XREF extraction (one Figure per XREF).

    Skips XREFs smaller than `min_dim x min_dim` px before aggregation —
    PyMuPDF sometimes lists vector-art outlines and 1-px separators as
    "images"; the floor strips that noise without dropping legitimate small
    panels (most composite cells are 100+ px on at least one axis).

    Returns one `Figure` per logical caption (or per XREF on caption-less
    pages), with caption text matched by figure number when possible.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    figures: list[Figure] = []
    paper_out = out_dir / paper_id
    seen_xrefs: set[int] = set()

    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_no = page.number + 1
            captions = _captions_with_bboxes(page)

            page_xrefs: list[_XrefRecord] = []
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                # img_info: (xref, smask, width, height, bpc, colorspace, alt, name, filter, ...).
                width, height = img_info[2], img_info[3]
                if width < min_dim or height < min_dim:
                    continue
                page_xrefs.append(
                    _XrefRecord(
                        xref=xref, bbox=_figure_bbox(page, xref), width=width, height=height
                    )
                )

            if not page_xrefs:
                continue

            by_caption, unassigned = _assign_xrefs_to_captions(page_xrefs, captions)
            page_figure_idx = 0

            # Caption-anchored figures: one Figure per non-empty caption group.
            for label, members in by_caption.items():
                if not members:
                    continue
                # Representative XREF = largest by area (the "main panel" view).
                # For a 5x2 of 4x4 thumb grid, the largest member is likely the
                # top-level wrapper image when present; otherwise an arbitrary
                # but consistent cell.
                rep = max(members, key=lambda r: r.width * r.height)
                page_figure_idx += 1
                figure_id = _safe_index(paper_id, page_no, page_figure_idx)
                image_path = paper_out / f"{_safe_filename(figure_id)}.png"
                try:
                    _save_pixmap(doc, rep.xref, image_path)
                except (RuntimeError, ValueError) as exc:
                    _log.warning(
                        "figure.save_failed",
                        paper_id=paper_id,
                        page=page_no,
                        xref=rep.xref,
                        error=str(exc),
                    )
                    continue
                member_bboxes = [r.bbox for r in members if r.bbox is not None]
                bbox = _union_bbox(member_bboxes) if member_bboxes else rep.bbox
                caption_text, _label_bbox = captions[label]
                if len(members) > 1:
                    _log.debug(
                        "figure.aggregated",
                        paper_id=paper_id,
                        figure_id=figure_id,
                        caption_label=label,
                        n_members=len(members),
                    )
                figures.append(
                    Figure(
                        figure_id=figure_id,
                        paper_id=paper_id,
                        page_number=page_no,
                        caption=caption_text,
                        image_path=image_path,
                        bbox=bbox,
                    )
                )

            # Fallback: XREFs not paired with any caption (no captions on the
            # page, or no caption had a bbox to anchor against). Emit one
            # Figure per XREF as before — this path is what handles title-page
            # logos, header graphics, and pages where caption detection failed.
            for record in unassigned:
                page_figure_idx += 1
                figure_id = _safe_index(paper_id, page_no, page_figure_idx)
                image_path = paper_out / f"{_safe_filename(figure_id)}.png"
                try:
                    _save_pixmap(doc, record.xref, image_path)
                except (RuntimeError, ValueError) as exc:
                    _log.warning(
                        "figure.save_failed",
                        paper_id=paper_id,
                        page=page_no,
                        xref=record.xref,
                        error=str(exc),
                    )
                    continue
                figures.append(
                    Figure(
                        figure_id=figure_id,
                        paper_id=paper_id,
                        page_number=page_no,
                        caption="",
                        image_path=image_path,
                        bbox=record.bbox,
                    )
                )
    return figures
