"""Figure/table → Chunk converters."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.chunking import figure_to_chunk, table_to_chunk
from src.types import Bbox, Figure, Table


def test_figure_to_chunk_uses_pdf_caption_when_no_vlm() -> None:
    fig = Figure(
        figure_id="p1::p3::fig1",
        paper_id="p1",
        page_number=3,
        caption="Figure 1: Architecture overview.",
        image_path=Path("data/figures/p1/p1__p3__fig1.png"),
    )
    chunk = figure_to_chunk(fig)
    assert chunk.text == "Figure 1: Architecture overview."
    assert chunk.metadata["kind"] == "figure"
    assert chunk.metadata["has_vlm_caption"] is False
    assert chunk.chunk_id == "p1::p3::fig1"
    assert chunk.page_numbers == [3]


def test_figure_to_chunk_prefers_vlm_caption_when_set() -> None:
    """VLM caption wins over PDF caption — concatenating both was empirically worse."""
    fig = Figure(
        figure_id="p1::p3::fig1",
        paper_id="p1",
        page_number=3,
        caption="Figure 1: Architecture overview.",
        image_path=Path("dummy.png"),
        vlm_caption="A two-column architecture diagram with an encoder on the left and decoder on the right.",
    )
    chunk = figure_to_chunk(fig)
    assert chunk.text.startswith("A two-column")
    assert "Architecture overview" not in chunk.text  # PDF caption deliberately not included
    assert chunk.metadata["has_vlm_caption"] is True


def test_figure_to_chunk_uses_only_vlm_when_pdf_caption_missing() -> None:
    fig = Figure(
        figure_id="p1::p3::fig1",
        paper_id="p1",
        page_number=3,
        caption="",
        image_path=Path("dummy.png"),
        vlm_caption="A schematic of two parallel attention heads.",
    )
    chunk = figure_to_chunk(fig)
    assert chunk.text == "A schematic of two parallel attention heads."


def test_figure_to_chunk_falls_back_to_id_when_no_caption() -> None:
    fig = Figure(
        figure_id="p1::p3::fig1",
        paper_id="p1",
        page_number=3,
        caption="",
        image_path=Path("dummy.png"),
    )
    chunk = figure_to_chunk(fig)
    assert chunk.text == "[p1::p3::fig1]"


def test_table_to_chunk_concatenates_caption_and_markdown() -> None:
    md = "| h1 | h2 |\n|---|---|\n| a | b |"
    tbl = Table(
        table_id="p1::p4::tab1",
        paper_id="p1",
        page_number=4,
        markdown=md,
        caption="Table 1: A small table.",
    )
    chunk = table_to_chunk(tbl)
    assert chunk.text.startswith("Table 1: A small table.")
    assert md in chunk.text
    assert chunk.metadata["kind"] == "table"


def test_table_to_chunk_handles_missing_caption() -> None:
    tbl = Table(
        table_id="p1::p4::tab1",
        paper_id="p1",
        page_number=4,
        markdown="| h |\n|---|\n| v |",
        caption=None,
    )
    chunk = table_to_chunk(tbl)
    assert chunk.text.startswith("| h |")
    assert chunk.metadata["kind"] == "table"


# ADR 0009: bbox propagation. figure_to_chunk / table_to_chunk pack bbox
# into metadata['bbox'] as a 4-list when the source has one; absent
# otherwise so downstream code can `.get("bbox")` without checking shape.


def test_figure_to_chunk_packs_bbox_into_metadata() -> None:
    fig = Figure(
        figure_id="p1::p3::fig1",
        paper_id="p1",
        page_number=3,
        caption="Figure 1: X.",
        image_path=Path("dummy.png"),
        bbox=Bbox(x0=72.0, y0=144.0, x1=540.0, y1=432.0),
    )
    chunk = figure_to_chunk(fig)
    assert chunk.metadata["bbox"] == [72.0, 144.0, 540.0, 432.0]


def test_figure_to_chunk_omits_bbox_when_none() -> None:
    fig = Figure(
        figure_id="p1::p3::fig1",
        paper_id="p1",
        page_number=3,
        caption="Figure 1: X.",
        image_path=Path("dummy.png"),
        bbox=None,
    )
    chunk = figure_to_chunk(fig)
    assert "bbox" not in chunk.metadata


def test_table_to_chunk_packs_bbox_into_metadata() -> None:
    tbl = Table(
        table_id="p1::p4::tab1",
        paper_id="p1",
        page_number=4,
        markdown="| h |\n|---|\n| v |",
        caption="Table 1: Y.",
        bbox=Bbox(x0=50.0, y0=100.0, x1=450.0, y1=300.0),
    )
    chunk = table_to_chunk(tbl)
    assert chunk.metadata["bbox"] == [50.0, 100.0, 450.0, 300.0]


def test_table_to_chunk_omits_bbox_when_none() -> None:
    tbl = Table(
        table_id="p1::p4::tab1",
        paper_id="p1",
        page_number=4,
        markdown="| h |\n|---|\n| v |",
        caption=None,
        bbox=None,
    )
    chunk = table_to_chunk(tbl)
    assert "bbox" not in chunk.metadata


# Bbox model validation


def test_bbox_rejects_inverted_x() -> None:
    with pytest.raises(ValueError, match=r"x1.*must be > x0"):
        Bbox(x0=100.0, y0=0.0, x1=50.0, y1=100.0)


def test_bbox_rejects_inverted_y() -> None:
    with pytest.raises(ValueError, match=r"y1.*must be > y0"):
        Bbox(x0=0.0, y0=100.0, x1=100.0, y1=50.0)


def test_bbox_rejects_zero_area() -> None:
    with pytest.raises(ValueError):
        Bbox(x0=10.0, y0=10.0, x1=10.0, y1=20.0)


def test_bbox_rejects_negative_coords() -> None:
    with pytest.raises(ValueError):
        Bbox(x0=-1.0, y0=0.0, x1=100.0, y1=100.0)


def test_bbox_as_list_round_trips() -> None:
    b = Bbox(x0=1.5, y0=2.5, x1=11.5, y1=22.5)
    assert b.as_list() == [1.5, 2.5, 11.5, 22.5]
    assert isinstance(b.as_list(), list)
    assert all(isinstance(v, float) for v in b.as_list())


def test_bbox_is_frozen() -> None:
    b = Bbox(x0=0.0, y0=0.0, x1=10.0, y1=10.0)
    with pytest.raises((TypeError, ValueError)):
        b.x0 = 5.0  # type: ignore[misc]
