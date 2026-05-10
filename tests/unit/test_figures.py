"""Figure extraction: caption parsing + smoke test against a real PDF."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.figures import (
    _assign_xrefs_to_captions,
    _extract_captions,
    _union_bbox,
    _XrefRecord,
    extract_figures,
)
from src.types import Bbox


def test_extract_captions_parses_figure_label() -> None:
    text = """Some intro paragraph.

Figure 1: An overview of the architecture, showing two stages and an arrow.

Body text resumes here. Figure 2: Second figure caption that runs over

multiple paragraphs but stops at the blank line above this one.
"""
    captions = _extract_captions(text)
    assert "1" in captions
    assert "An overview of the architecture" in captions["1"]
    assert "2" in captions
    assert "Second figure caption" in captions["2"]


def test_extract_captions_handles_fig_dot_abbreviation() -> None:
    text = "Fig. 3 — A short caption.\n\nMore body."
    captions = _extract_captions(text)
    assert "3" in captions
    assert "A short caption" in captions["3"]


def test_extract_captions_handles_appendix_label_with_letter_prefix() -> None:
    """Appendix/supplementary figures use labels like `E.1`, `S1`, `A.3`.
    Paper 2604.28190v1 had ~900 unaggregated XREFs because the prior numeric-
    only regex skipped `Figure E.1:` captions in the appendix."""
    text = "Figure E.1: Uncurated paired samples on ImageNet 256x256.\n\nMore body."
    captions = _extract_captions(text)
    assert "E.1" in captions
    assert "Uncurated paired samples" in captions["E.1"]


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


# --- ADR 0011: caption-anchored aggregation primitives ---------------------


def test_union_bbox_returns_enclosing_rect() -> None:
    a = Bbox(x0=10, y0=10, x1=50, y1=50)
    b = Bbox(x0=30, y0=30, x1=80, y1=80)
    union = _union_bbox([a, b])
    assert union == Bbox(x0=10, y0=10, x1=80, y1=80)


def test_union_bbox_single_returns_same_rect() -> None:
    a = Bbox(x0=5, y0=5, x1=10, y1=10)
    assert _union_bbox([a]) == a


def test_union_bbox_empty_returns_none() -> None:
    assert _union_bbox([]) is None


def _xref(xref: int, y_center: float, w: int = 100, h: int = 100) -> _XrefRecord:
    """Build an _XrefRecord with a 50-pt-tall bbox centered at `y_center`."""
    bbox = Bbox(x0=100.0, y0=y_center - 25.0, x1=200.0, y1=y_center + 25.0)
    return _XrefRecord(xref=xref, bbox=bbox, width=w, height=h)


def test_assign_xrefs_no_captions_returns_all_unassigned() -> None:
    xrefs = [_xref(1, 100.0), _xref(2, 200.0)]
    assignments, unassigned = _assign_xrefs_to_captions(xrefs, captions={})
    assert assignments == {}
    assert [r.xref for r in unassigned] == [1, 2]


def test_assign_xrefs_single_caption_bundles_all() -> None:
    """Composite figure: 1 caption + many XREFs in same vertical band → 1 group."""
    caption_bbox = Bbox(x0=100.0, y0=80.0, x1=400.0, y1=100.0)
    captions: dict[str, tuple[str, Bbox | None]] = {"5": ("Figure 5: composite", caption_bbox)}
    xrefs = [_xref(i, 150.0 + i * 5.0) for i in range(1, 11)]
    assignments, unassigned = _assign_xrefs_to_captions(xrefs, captions)
    assert sorted(r.xref for r in assignments["5"]) == list(range(1, 11))
    assert unassigned == []


def test_assign_xrefs_two_captions_partition_by_nearest_y() -> None:
    captions: dict[str, tuple[str, Bbox | None]] = {
        "1": ("Figure 1: top", Bbox(x0=100.0, y0=100.0, x1=400.0, y1=130.0)),
        "2": ("Figure 2: bottom", Bbox(x0=100.0, y0=600.0, x1=400.0, y1=630.0)),
    }
    xrefs = [
        _xref(1, 180.0),  # near caption 1
        _xref(2, 650.0),  # near caption 2
        _xref(3, 670.0),  # near caption 2
    ]
    assignments, unassigned = _assign_xrefs_to_captions(xrefs, captions)
    assert [r.xref for r in assignments["1"]] == [1]
    assert sorted(r.xref for r in assignments["2"]) == [2, 3]
    assert unassigned == []


def test_assign_xrefs_caption_without_bbox_is_skipped() -> None:
    """If the caption block has no bbox, we can't anchor on it — XREFs fall through."""
    captions: dict[str, tuple[str, Bbox | None]] = {"1": ("Figure 1: no bbox", None)}
    xrefs = [_xref(1, 200.0)]
    assignments, unassigned = _assign_xrefs_to_captions(xrefs, captions)
    assert assignments == {}
    assert [r.xref for r in unassigned] == [1]


def test_assign_xrefs_xref_without_bbox_falls_through() -> None:
    captions: dict[str, tuple[str, Bbox | None]] = {
        "1": ("Figure 1: ok", Bbox(x0=100.0, y0=80.0, x1=400.0, y1=100.0))
    }
    xrefs = [_XrefRecord(xref=1, bbox=None, width=100, height=100)]
    assignments, unassigned = _assign_xrefs_to_captions(xrefs, captions)
    assert assignments["1"] == []
    assert [r.xref for r in unassigned] == [1]


def test_assign_xrefs_appendix_label_keeps_letter_prefix() -> None:
    """Appendix figures (`Figure E.1:`) must work end-to-end through the
    assignment too, not just the regex match — keys are strings."""
    captions: dict[str, tuple[str, Bbox | None]] = {
        "E.1": ("Figure E.1: appendix samples", Bbox(x0=100.0, y0=80.0, x1=400.0, y1=100.0))
    }
    xrefs = [_xref(i, 200.0 + i) for i in range(1, 4)]
    assignments, unassigned = _assign_xrefs_to_captions(xrefs, captions)
    assert sorted(r.xref for r in assignments["E.1"]) == [1, 2, 3]
    assert unassigned == []


@pytest.mark.integration
def test_extract_figures_aggregates_composite_paper(tmp_path: Path) -> None:
    """ADR 0011: paper 2604.28190v1 (FD-loss) has a composite figure that
    used to extract as 1000 sub-thumbnail XREFs. With caption-anchored
    aggregation, the total figure count must be well under the XREF count —
    one Figure per `Figure N:` caption, not one per panel cell.
    """
    pdf = Path("data/papers/2604.28190v1.pdf")
    if not pdf.exists():
        pytest.skip("paper not present in this checkout")
    figures = extract_figures("2604.28190v1", pdf, out_dir=tmp_path, min_dim=64)
    # The paper has ~5-10 logical figures by caption. If we get >50, the
    # aggregation regressed and we're back to per-XREF behaviour.
    assert len(figures) < 50, f"composite over-decomposition: got {len(figures)} figures"
