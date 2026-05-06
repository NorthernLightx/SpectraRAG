"""Figure/table → Chunk converters."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.chunking import figure_to_chunk, table_to_chunk
from src.types import Figure, Table


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
