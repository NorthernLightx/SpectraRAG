"""Figure extraction: caption parsing + smoke test against a real PDF."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.figures import _extract_captions, extract_figures


def test_extract_captions_parses_figure_label() -> None:
    text = """Some intro paragraph.

Figure 1: An overview of the architecture, showing two stages and an arrow.

Body text resumes here. Figure 2: Second figure caption that runs over

multiple paragraphs but stops at the blank line above this one.
"""
    captions = _extract_captions(text)
    assert 1 in captions
    assert "An overview of the architecture" in captions[1]
    assert 2 in captions
    assert "Second figure caption" in captions[2]


def test_extract_captions_handles_fig_dot_abbreviation() -> None:
    text = "Fig. 3 — A short caption.\n\nMore body."
    captions = _extract_captions(text)
    assert 3 in captions
    assert "A short caption" in captions[3]


def test_extract_captions_returns_empty_when_no_labels() -> None:
    assert _extract_captions("Just body text with no figures.") == {}


@pytest.mark.integration
def test_extract_figures_against_real_paper(tmp_path: Path) -> None:
    """Smoke test: 2604.22753v1 has at least one extractable embedded image."""
    pdf = Path("data/papers/2604.22753v1.pdf")
    if not pdf.exists():
        pytest.skip("paper not present in this checkout")
    figures = extract_figures("2604.22753v1", pdf, out_dir=tmp_path)
    # Most ArXiv ML papers have at least one figure; if the extractor returns
    # zero, that's a regression worth catching.
    assert isinstance(figures, list)
    for fig in figures:
        assert fig.image_path.exists()
        assert fig.paper_id == "2604.22753v1"
        assert fig.figure_id.startswith("2604.22753v1::")


@pytest.mark.integration
def test_extract_figures_captures_bbox_when_available(tmp_path: Path) -> None:
    """ADR 0009: at least one extracted figure should have a bbox.

    On the 20-paper v3 corpus, the vast majority of embedded images are
    placed via Image XObject and PyMuPDF's `get_image_rects` returns a
    non-empty list. If *every* extracted figure's bbox is None for a paper
    with multiple figures, that's a regression in the extraction path.
    """
    pdf = Path("data/papers/2604.22753v1.pdf")
    if not pdf.exists():
        pytest.skip("paper not present in this checkout")
    figures = extract_figures("2604.22753v1", pdf, out_dir=tmp_path)
    if not figures:
        pytest.skip("no figures extracted from this paper")
    bboxed = [f for f in figures if f.bbox is not None]
    assert bboxed, "expected at least one figure to have a bbox; ADR 0009 §Failure modes"
    # Each captured bbox should be sane: positive area, within reasonable
    # PDF page dimensions (US Letter at 72 dpi = 612 x 792 points; ArXiv
    # papers vary but rarely exceed 1500 points on either axis).
    for f in bboxed:
        assert f.bbox is not None  # narrow for mypy
        assert f.bbox.x1 > f.bbox.x0
        assert f.bbox.y1 > f.bbox.y0
        assert 0 <= f.bbox.x0 < 1500
        assert 0 <= f.bbox.y0 < 2000
        assert (f.bbox.x1 - f.bbox.x0) > 1.0  # non-degenerate width
        assert (f.bbox.y1 - f.bbox.y0) > 1.0  # non-degenerate height
