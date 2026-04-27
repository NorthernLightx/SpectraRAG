"""PDF extraction: pages produce Page objects with text in reading order."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from src.ingestion.pdf import extract_pages
from src.types import Page


def _make_pdf(tmp_path: Path, pages_text: list[str]) -> Path:
    """Generate a synthetic PDF for tests. No external fixtures."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    pdf_path = tmp_path / "tiny.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_extract_pages_yields_one_page_per_pdf_page(tmp_path: Path) -> None:
    pdf_path = _make_pdf(tmp_path, ["First page text.", "Second page text."])

    pages = extract_pages(paper_id="p1", pdf_path=pdf_path)

    assert len(pages) == 2
    assert all(isinstance(p, Page) for p in pages)
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2
    assert "First page text." in pages[0].text
    assert "Second page text." in pages[1].text
    assert all(p.paper_id == "p1" for p in pages)


def test_extract_pages_strips_whitespace_only_pages(tmp_path: Path) -> None:
    pdf_path = _make_pdf(tmp_path, ["Real text.", "   "])
    pages = extract_pages(paper_id="p1", pdf_path=pdf_path)
    assert len(pages) == 2
    assert pages[1].text.strip() == ""


def test_extract_pages_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_pages(paper_id="p1", pdf_path=tmp_path / "missing.pdf")
