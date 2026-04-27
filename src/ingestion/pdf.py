"""PDF page extraction via PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import fitz

from src.types import Page


def extract_pages(paper_id: str, pdf_path: Path) -> list[Page]:
    """Open PDF and extract one Page per physical page, in reading order."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[Page] = []
    with fitz.open(pdf_path) as doc:
        for index, fitz_page in enumerate(doc):
            text = fitz_page.get_text("text") or ""
            pages.append(Page(paper_id=paper_id, page_number=index + 1, text=text, image_path=None))
    return pages
