"""Section-aware chunking: respects section boundaries, falls back when none found."""

from __future__ import annotations

from src.ingestion.chunking import chunk_pages, split_into_sections
from src.types import Page

_PAPER_BODY = """\
1 Introduction

This is the introduction. It contains background information about the topic.
The introduction has multiple sentences explaining motivation.

2 Related Work

Prior work in the field has explored several directions. We summarise the
relevant strands here, citing key references.

3 Method

Our method consists of three steps. First, we encode inputs. Second, we
retrieve. Third, we generate.

3.1 Encoding

Encoding uses a transformer.

4 Conclusion

We presented a method.
"""


def test_split_into_sections_identifies_numbered_headings() -> None:
    sections = split_into_sections(_PAPER_BODY)
    titles = [s.title for s in sections]
    assert "1 Introduction" in titles
    assert "2 Related Work" in titles
    assert "3 Method" in titles
    assert "3.1 Encoding" in titles
    assert "4 Conclusion" in titles
    assert sections[0].title == "1 Introduction"
    assert "background information" in sections[0].text


def test_split_into_sections_falls_back_when_no_headings() -> None:
    sections = split_into_sections("Plain text with no section headings at all.")
    assert len(sections) == 1
    assert sections[0].title is None
    assert "Plain text" in sections[0].text


def test_chunk_pages_carries_section_in_metadata() -> None:
    page = Page(paper_id="p1", page_number=1, text=_PAPER_BODY)
    chunks = chunk_pages([page], target_chars=200, overlap_chars=40)

    assert len(chunks) >= 5
    intro_chunks = [c for c in chunks if c.section == "1 Introduction"]
    assert len(intro_chunks) >= 1
    assert all(c.paper_id == "p1" for c in chunks)
    assert all(c.page_numbers == [1] for c in chunks)
    assert all(c.chunk_id.startswith("p1::") for c in chunks)


def test_chunk_pages_respects_size_budget() -> None:
    page = Page(paper_id="p1", page_number=1, text=_PAPER_BODY)
    chunks = chunk_pages([page], target_chars=200, overlap_chars=40)
    assert all(len(c.text) <= 200 + 40 + 50 for c in chunks)


def test_chunk_pages_handles_empty_input() -> None:
    assert chunk_pages([], target_chars=200, overlap_chars=40) == []
