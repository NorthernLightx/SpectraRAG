"""Page-image rendering for the visual retrieval path.

ColPali-style retrievers (ColQwen2, ColPali, etc.) embed *whole pages* as
images rather than working from extracted text. This module renders each
PDF page to a PNG at a configurable DPI and returns the path list, leaving
embedding to the visual retriever.

We render at 150 DPI by default — a balance between fidelity (text legible
to the VLM) and ColQwen2's input pixel budget (it resizes to a fixed grid
regardless, but we want the source crisp enough that resampling preserves
small text).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz

from src.observability.logging import get_logger

_log = get_logger(__name__)

_DEFAULT_DPI = 150


@dataclass(frozen=True)
class RenderedPage:
    """A page rendered to disk for visual retrieval."""

    paper_id: str
    page_number: int
    image_path: Path


def _safe_filename(paper_id: str, page_no: int) -> str:
    """Windows-safe page filename. Mirrors `figures._safe_filename` convention."""
    return f"{paper_id}_p{page_no}.png".replace(":", "_")


def render_pages(
    paper_id: str,
    pdf_path: Path,
    *,
    out_dir: Path = Path("data/pages"),
    dpi: int = _DEFAULT_DPI,
) -> list[RenderedPage]:
    """Render each page of `pdf_path` to a PNG under `out_dir/<paper_id>/`.

    Idempotent — if the target file already exists with non-zero size, it's
    treated as already rendered and skipped (useful for re-runs of the same
    paper).
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    paper_dir = out_dir / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)

    # PyMuPDF's `Matrix(dpi/72, dpi/72)` is the standard PDF→pixmap zoom ratio.
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    rendered: list[RenderedPage] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_no = page.number + 1
            image_path = paper_dir / _safe_filename(paper_id, page_no)
            if image_path.exists() and image_path.stat().st_size > 0:
                rendered.append(
                    RenderedPage(paper_id=paper_id, page_number=page_no, image_path=image_path)
                )
                continue
            try:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                pix.save(str(image_path))
            except (RuntimeError, ValueError) as exc:
                _log.warning(
                    "page_render.failed",
                    paper_id=paper_id,
                    page=page_no,
                    error=str(exc),
                )
                continue
            rendered.append(
                RenderedPage(paper_id=paper_id, page_number=page_no, image_path=image_path)
            )
    return rendered
