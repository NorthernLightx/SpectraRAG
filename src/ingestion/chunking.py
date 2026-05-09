"""Section-aware chunking. Splits page text on numbered headings, then by char budget.

Also provides Figure/Table → Chunk converters: figures and tables are
first-class chunks in the same retrieval corpus, with `metadata['kind']` set
to "figure" / "table" so callers can distinguish them from text chunks at
display time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.types import Chunk, Figure, Page, Table

_SECTION_HEADING_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z][A-Za-z\s\-:&]{1,60})\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Section:
    """A logical section of a paper."""

    title: str | None
    text: str


def split_into_sections(body: str) -> list[Section]:
    """Split body text on numbered headings (e.g. '1 Introduction', '3.1 Encoding').

    If no headings are found, returns one untitled Section with the whole body.
    """
    matches = list(_SECTION_HEADING_RE.finditer(body))
    if not matches:
        return [Section(title=None, text=body.strip())]

    sections: list[Section] = []
    if matches[0].start() > 0:
        prelude = body[: matches[0].start()].strip()
        if prelude:
            sections.append(Section(title=None, text=prelude))

    for index, match in enumerate(matches):
        title = f"{match.group(1)} {match.group(2).strip()}"
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        text = body[body_start:body_end].strip()
        if text:
            sections.append(Section(title=title, text=text))
    return sections


def _windowed(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Sliding-window split with overlap. Breaks on sentence boundaries when possible."""
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]

    windows: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + target_chars, len(text))
        if end < len(text):
            slice_ = text[start:end]
            last_dot = max(slice_.rfind(". "), slice_.rfind("\n"))
            if last_dot > target_chars - 80:
                end = start + last_dot + 1
        windows.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return [w for w in windows if w]


def chunk_pages(
    pages: list[Page], *, target_chars: int = 1200, overlap_chars: int = 200
) -> list[Chunk]:
    """Turn a list of Pages into a list of Chunks, section-aware."""
    chunks: list[Chunk] = []
    counter = 0
    for page in pages:
        sections = split_into_sections(page.text)
        for section in sections:
            for window in _windowed(section.text, target_chars, overlap_chars):
                chunk_id = f"{page.paper_id}::p{page.page_number}::c{counter}"
                counter += 1
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        paper_id=page.paper_id,
                        page_numbers=[page.page_number],
                        text=window,
                        section=section.title,
                    )
                )
    return chunks


def figure_to_chunk(figure: Figure) -> Chunk:
    """Convert a Figure into a retrievable Chunk.

    Picks the *single best* caption source: VLM caption when present (it tends
    to add visual structure beyond the dense terminology PDFs capture), else
    the PDF-extracted caption. Empirically, concatenating both *hurts* — the
    longer combined text fooled the reranker into surfacing weak figure chunks
    over the strong text chunks that actually answer the query.

    Figures with no caption at all become a stub chunk with just the figure id —
    better than dropping them, since BM25 might still match the id when an
    answer cites a figure.

    `bbox` is packed into `metadata['bbox']` as `[x0, y0, x1, y1]` floats when
    figure extraction captured one (ADR 0009). Citation surface picks it up
    via `Generator._extract_citations` so demos can render region-precise
    highlights on the page image.
    """
    primary = (
        figure.vlm_caption
        if (figure.vlm_caption and figure.vlm_caption.strip())
        else (
            figure.caption if figure.caption and figure.caption.strip() else f"[{figure.figure_id}]"
        )
    )
    metadata: dict[str, object] = {
        "kind": "figure",
        "image_path": str(figure.image_path),
        "has_vlm_caption": figure.vlm_caption is not None,
    }
    if figure.bbox is not None:
        metadata["bbox"] = figure.bbox.as_list()
    return Chunk(
        chunk_id=figure.figure_id,
        paper_id=figure.paper_id,
        page_numbers=[figure.page_number],
        text=primary,
        section=None,
        metadata=metadata,
    )


def table_to_chunk(table: Table) -> Chunk:
    """Convert a Table into a retrievable Chunk.

    `text` is `caption\\n\\n<markdown>` so both caption-keyword queries and
    cell-content queries match. The markdown is preserved verbatim for display.

    `bbox` is packed into `metadata['bbox']` as `[x0, y0, x1, y1]` floats when
    PyMuPDF located the table on the page (ADR 0009).
    """
    parts = [p for p in (table.caption, table.markdown) if p]
    metadata: dict[str, object] = {"kind": "table"}
    if table.bbox is not None:
        metadata["bbox"] = table.bbox.as_list()
    return Chunk(
        chunk_id=table.table_id,
        paper_id=table.paper_id,
        page_numbers=[table.page_number],
        text="\n\n".join(parts) if parts else f"[{table.table_id}]",
        section=None,
        metadata=metadata,
    )
