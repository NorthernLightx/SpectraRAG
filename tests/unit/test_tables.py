"""Table extraction: cell-to-markdown rendering + caption parsing."""

from __future__ import annotations

from src.ingestion.tables import _cells_to_markdown, _extract_captions


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
