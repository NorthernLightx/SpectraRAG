"""Figure extraction from PDFs (Phase 2 — pipeline multi-modal).

PyMuPDF gives us two views of figures:
- Embedded image streams (via `page.get_images()`) — the bytes that the PDF
  embeds. Unique-by-XREF, so the same logo across pages is one figure not N.
- Rendered drawings (vector art) — not exported here; covered later if needed.

Caption association is heuristic: we scan the page text for `Figure N:` /
`Fig. N:` lines and pair them with the N-th image extracted from that page,
ordered top-to-bottom. Imperfect but adequate for ArXiv ML papers where the
caption is almost always immediately above or below the image.

Each Figure gets `caption` set to the PDF-extracted text. `vlm_caption` is
left None at this layer; a later pass can fill it from a vision model.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from src.observability.logging import get_logger
from src.types import Figure

_log = get_logger(__name__)

# Match `Figure 12: caption…` and `Fig. 3 — caption…`. The body runs until the
# next blank line (paragraph break) or another Figure/Table label, whichever
# comes first. `[\s\S]` is "any char including newlines" without flipping the
# whole pattern into DOTALL mode.
_FIG_LABEL_RE = re.compile(
    r"(?:Figure|Fig\.?)\s+(\d+)\s*[:.\-—]\s*([\s\S]*?)"
    r"(?=\n\s*\n|\n\s*(?:Figure|Fig\.?|Table)\s+\d+\s*[:.\-—]|\Z)",
    re.IGNORECASE,
)


def _safe_index(paper_id: str, page_no: int, idx: int) -> str:
    return f"{paper_id}::p{page_no}::fig{idx}"


def _safe_filename(figure_id: str) -> str:
    """Convert `paper::p1::fig1` into a Windows-safe filename — `:` is not legal there."""
    return figure_id.replace(":", "_")


def _extract_captions(page_text: str) -> dict[int, str]:
    """Return `{figure_number: caption_body}` parsed from page text."""
    captions: dict[int, str] = {}
    for match in _FIG_LABEL_RE.finditer(page_text):
        try:
            number = int(match.group(1))
        except ValueError:
            continue
        body = " ".join(match.group(2).split())
        # Truncate at the first paragraph break (blank line) — long captions
        # are often followed by body text we don't want to swallow.
        if "\n\n" in body:
            body = body.split("\n\n", 1)[0].strip()
        captions[number] = f"Figure {number}: {body}".strip()
    return captions


def _save_pixmap(doc: fitz.Document, xref: int, out_path: Path) -> None:
    """Render an XREF as a PNG, converting to RGB if needed (alpha + CMYK both fail save)."""
    pix = fitz.Pixmap(doc, xref)
    if pix.alpha or pix.colorspace is None or pix.colorspace.name != "DeviceRGB":
        pix = fitz.Pixmap(fitz.csRGB, pix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))


def extract_figures(
    paper_id: str,
    pdf_path: Path,
    *,
    out_dir: Path = Path("data/figures"),
    min_dim: int = 64,
) -> list[Figure]:
    """Extract embedded figures from `pdf_path`, save PNGs under `out_dir/<paper_id>/`.

    Skips images smaller than `min_dim x min_dim` px (logos, decorative bullets).
    Returns one `Figure` per saved image, with caption text matched by figure
    number when possible (else empty string).
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    figures: list[Figure] = []
    paper_out = out_dir / paper_id
    seen_xrefs: set[int] = set()

    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_no = page.number + 1
            captions = _extract_captions(page.get_text("text") or "")
            page_figure_idx = 0
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                # img_info: (xref, smask, width, height, bpc, colorspace, alt, name, filter, ...)
                width, height = img_info[2], img_info[3]
                if width < min_dim or height < min_dim:
                    continue
                page_figure_idx += 1
                figure_id = _safe_index(paper_id, page_no, page_figure_idx)
                image_path = paper_out / f"{_safe_filename(figure_id)}.png"
                try:
                    _save_pixmap(doc, xref, image_path)
                except (RuntimeError, ValueError) as exc:
                    _log.warning(
                        "figure.save_failed",
                        paper_id=paper_id,
                        page=page_no,
                        xref=xref,
                        error=str(exc),
                    )
                    continue
                # Pair with caption N if present; otherwise empty caption.
                caption = captions.get(page_figure_idx, "")
                figures.append(
                    Figure(
                        figure_id=figure_id,
                        paper_id=paper_id,
                        page_number=page_no,
                        caption=caption,
                        image_path=image_path,
                    )
                )
    return figures
