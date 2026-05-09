"""Table extraction: cell-to-markdown rendering + caption parsing."""

from __future__ import annotations

from dataclasses import dataclass

from src.ingestion.tables import _cells_to_markdown, _extract_captions, _table_bbox


def test_cells_to_markdown_renders_header_and_body() -> None:
    cells: list[list[str | None]] = [
        ["Method", "nDCG", "MRR"],
        ["Hybrid", "0.52", "0.50"],
        ["+ Rerank", "0.82", "0.88"],
    ]
    md = _cells_to_markdown(cells)
    lines = md.splitlines()
    assert lines[0] == "| Method | nDCG | MRR |"
    assert lines[1] == "|---|---|---|"
    assert lines[2] == "| Hybrid | 0.52 | 0.50 |"
    assert lines[3] == "| + Rerank | 0.82 | 0.88 |"


def test_cells_to_markdown_pads_short_rows() -> None:
    cells: list[list[str | None]] = [["a", "b", "c"], ["x"]]
    md = _cells_to_markdown(cells)
    # Row of length 1 is padded to width 3.
    assert md.splitlines()[2] == "| x |   |   |"


def test_cells_to_markdown_replaces_none_with_space() -> None:
    cells: list[list[str | None]] = [["h1", "h2"], [None, "v"]]
    md = _cells_to_markdown(cells)
    assert "|   | v |" in md


def test_cells_to_markdown_flattens_multiline_cells() -> None:
    cells: list[list[str | None]] = [["h"], ["one\ntwo\nthree"]]
    md = _cells_to_markdown(cells)
    assert "| one two three |" in md


def test_cells_to_markdown_returns_empty_for_empty_input() -> None:
    assert _cells_to_markdown([]) == ""


def test_extract_table_captions() -> None:
    text = """Body.

Table 1: Comparison of methods on the held-out test set.

More body. Table 2 — Per-task breakdown of recall.
"""
    captions = _extract_captions(text)
    assert "Comparison of methods" in captions[1]
    assert "Per-task breakdown" in captions[2]


# ADR 0009: _table_bbox handles PyMuPDF's various bbox shapes (Rect, tuple)
# and degrades gracefully on absent/degenerate values rather than crashing
# the ingestion pass.


@dataclass
class _StubFound:
    bbox: object


def test_table_bbox_from_4_tuple() -> None:
    bbox = _table_bbox(_StubFound(bbox=(50.0, 100.0, 450.0, 300.0)))
    assert bbox is not None
    assert bbox.as_list() == [50.0, 100.0, 450.0, 300.0]


def test_table_bbox_from_rect_like_iterable() -> None:
    """PyMuPDF Rects are 4-iterables of floats; tuple coercion handles them."""
    bbox = _table_bbox(_StubFound(bbox=[10.0, 20.0, 110.0, 220.0]))
    assert bbox is not None
    assert bbox.as_list() == [10.0, 20.0, 110.0, 220.0]


def test_table_bbox_returns_none_for_missing_attr() -> None:
    @dataclass
    class _NoBbox:
        pass

    assert _table_bbox(_NoBbox()) is None


def test_table_bbox_returns_none_for_explicit_none() -> None:
    assert _table_bbox(_StubFound(bbox=None)) is None


def test_table_bbox_returns_none_for_wrong_length() -> None:
    assert _table_bbox(_StubFound(bbox=(10.0, 20.0, 30.0))) is None
    assert _table_bbox(_StubFound(bbox=(10.0, 20.0, 30.0, 40.0, 50.0))) is None


def test_table_bbox_returns_none_for_non_numeric() -> None:
    assert _table_bbox(_StubFound(bbox=("x", "y", "z", "w"))) is None


def test_table_bbox_returns_none_for_degenerate_rect() -> None:
    # x1 == x0 → zero width → Bbox validator rejects → _table_bbox returns None.
    assert _table_bbox(_StubFound(bbox=(50.0, 100.0, 50.0, 200.0))) is None
    assert _table_bbox(_StubFound(bbox=(50.0, 100.0, 100.0, 100.0))) is None
