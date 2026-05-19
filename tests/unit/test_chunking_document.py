"""Document-level chunking behaviour added in ADR 0017.

The pre-existing section/size/empty contracts live in `test_chunking.py`;
these cover the new cross-page, header-strip, and soup-drop behaviour.
"""

from __future__ import annotations

from src.ingestion.chunking import chunk_pages
from src.types import Page


def _pages(texts: list[str], paper_id: str = "p1") -> list[Page]:
    return [Page(paper_id=paper_id, page_number=i + 1, text=t) for i, t in enumerate(texts)]


def test_chunk_spans_page_boundary() -> None:
    # One untitled section flowing across the page break → at least one chunk
    # must carry both page numbers (the per-page chunker could not do this).
    page = " ".join(f"word{i}" for i in range(120))
    chunks = chunk_pages(_pages([page, page]), target_chars=200, overlap_chars=60)
    assert any(c.page_numbers == [1, 2] for c in chunks)


def test_running_header_stripped_from_chunks() -> None:
    pages = _pages(
        [
            "Running Head\nIntroduction text that is reasonably long to form a chunk body.",
            "Running Head\nMethod text that is also reasonably long to form a chunk body.",
            "Running Head\nResults text that is also reasonably long to form a chunk body.",
        ]
    )
    chunks = chunk_pages(pages, target_chars=200, overlap_chars=40)
    assert chunks
    assert all(not c.text.startswith("Running Head") for c in chunks)


def test_number_soup_page_yields_no_chunks() -> None:
    grid = "64 128 256 384 512 0.000122 0.000173 2.127 2.126 2.134 2.142 2.150 2.161"
    assert chunk_pages(_pages([grid])) == []


def test_section_isolated_soup_is_dropped_prose_kept() -> None:
    # A section whose body is entirely number soup contributes no chunk;
    # prose sections survive. (A window straddling a prose/soup boundary with
    # no heading between is mixed and kept — dropping it would lose content.)
    pages = _pages(
        [
            "1 Method\nThe approach encodes inputs with a transformer and retrieves "
            "relevant passages before the generation step produces an answer.",
            "2 Data\n0.1 0.2 0.3 64 128 256 2.127 2.126 2.134 2.142 2.150 2.161 2.178",
        ]
    )
    chunks = chunk_pages(pages, target_chars=200, overlap_chars=40)
    joined = " ".join(c.text for c in chunks)
    assert "transformer" in joined
    assert all(c.section != "2 Data" for c in chunks)
    assert "2.127 2.126 2.134" not in joined
