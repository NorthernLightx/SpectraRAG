"""Document-level, structure-aware chunking (ADR 0017).

Chunks are cut from the *whole document* (header-stripped pages concatenated
with a char→page map), not page by page. This keeps a section that spans a
page break as one unit and stops cutting paragraphs mid-sentence at page
boundaries.

Two noise classes are removed here (ADR 0017, evidence in
`scripts/experiments/quantify_corpus_junk.py`):

- running headers / page numbers — stripped per page in `clean.py` before
  concatenation (PyMuPDF emits them as their own lines).
- figure/table interior "number soup" — dropped per window via
  `clean.is_soup`.

Bibliography removal is *not* done here. Lexical heuristics cannot robustly
separate a reference list from citation-dense body/appendix text on this
corpus (the introduction of 2604.22753v1 cites more years-per-char than its
own reference list — see `quantify_corpus_junk.py`); doing it by region
excision destroyed golden-anchored appendix content. ADR 0017 defers it to
the GraphRAG ingestion pass (Step 1), which already runs an LLM over every
chunk and can judge "is this a reference list" reliably.

Also provides Figure/Table → Chunk converters: figures and tables are
first-class chunks in the same retrieval corpus, with `metadata['kind']` set
to "figure" / "table" so callers can distinguish them from text chunks at
display time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.ingestion.clean import detect_running_header, is_soup, strip_page_furniture
from src.observability.logging import get_logger
from src.types import Chunk, Figure, Page, Table

_log = get_logger(__name__)

# `1 Introduction`, `3.1 Encoding` — numbered headings on their own line.
_NUMBERED_HEADING_RE = re.compile(
    r"^[ \t]*(\d+(?:\.\d+)*)[ \t]+([A-Z][A-Za-z][A-Za-z\s\-:&]{1,60})[ \t]*$",
    re.MULTILINE,
)
# `A Use of LLMs`, `B.1 Task Collection`, `Appendix C Derivations` — appendix
# sections use a letter index instead of a number. Same conservative title
# shape as numbered headings so body lines are not mistaken for headings.
_APPENDIX_HEADING_RE = re.compile(
    r"^[ \t]*(?:Appendix[ \t]+)?([A-Z](?:\.\d+)*)[ \t]+([A-Z][A-Za-z][A-Za-z\s\-:&]{1,60})[ \t]*$",
    re.MULTILINE,
)
# Standalone named sections (own line, optional trailing colon, any case).
# "References"/"Bibliography" stay here so the section is *labelled* (useful
# for the Step-1 LLM filter and for section metadata); it is not dropped.
_NAMED_HEADING_RE = re.compile(
    r"^[ \t]*(Abstract|Acknowledge?ments?|Conclusions?|References|Bibliography"
    r"|Appendix|Supplementary(?:[ \t]+Material)?)[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class Section:
    """A logical section of a paper."""

    title: str | None
    text: str


def _heading_spans(text: str) -> list[tuple[int, str]]:
    """Sorted `(offset, title)` of every numbered / appendix / named heading."""
    seen: dict[int, str] = {}
    for rx, numbered in (
        (_NUMBERED_HEADING_RE, True),
        (_APPENDIX_HEADING_RE, True),
        (_NAMED_HEADING_RE, False),
    ):
        for m in rx.finditer(text):
            start = m.start()
            if start in seen:
                continue
            title = f"{m.group(1)} {m.group(2).strip()}" if numbered else m.group(1).strip()
            seen[start] = " ".join(title.split())
    return sorted(seen.items())


def _section_spans(doc: str) -> list[tuple[int, int, str | None]]:
    """`(body_start, body_end, title)` per section, in `doc` char coordinates.

    `body_start` is the char after the heading's own line (headings are
    full-line, regex-anchored), so the heading text is excluded from the body
    — same as the pre-ADR-0017 behaviour. Text before the first heading is an
    untitled prelude. Offsets are exact (no string search) so windows map
    back to pages precisely.
    """
    heads = _heading_spans(doc)
    if not heads:
        return [(0, len(doc), None)] if doc.strip() else []

    spans: list[tuple[int, int, str | None]] = []
    if doc[: heads[0][0]].strip():
        spans.append((0, heads[0][0], None))
    for index, (start, title) in enumerate(heads):
        newline = doc.find("\n", start)
        body_start = newline + 1 if newline != -1 else len(doc)
        body_end = heads[index + 1][0] if index + 1 < len(heads) else len(doc)
        if body_start < body_end:
            spans.append((body_start, body_end, title))
    return spans


def split_into_sections(body: str) -> list[Section]:
    """Split `body` on headings (numbered, appendix, or named).

    If no heading is found, returns one untitled Section with the whole body.
    """
    sections = [
        Section(title=title, text=text)
        for start, end, title in _section_spans(body)
        if (text := body[start:end].strip())
    ]
    return sections or [Section(title=None, text=body.strip())]


def _window_spans(text: str, target_chars: int, overlap_chars: int) -> list[tuple[int, int]]:
    """Sliding-window `(start, end)` spans with overlap, breaking on sentences."""
    if not text:
        return []
    if len(text) <= target_chars:
        return [(0, len(text))]

    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + target_chars, len(text))
        if end < len(text):
            slice_ = text[start:end]
            last_dot = max(slice_.rfind(". "), slice_.rfind("\n"))
            if last_dot > target_chars - 80:
                end = start + last_dot + 1
        spans.append((start, end))
        if end == len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return spans


def _windowed(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Sliding-window split with overlap. Breaks on sentence boundaries when possible."""
    out = [text[a:b].strip() for a, b in _window_spans(text, target_chars, overlap_chars)]
    return [w for w in out if w]


def chunk_pages(
    pages: list[Page], *, target_chars: int = 1200, overlap_chars: int = 200
) -> list[Chunk]:
    """Turn Pages into Chunks: header-stripped, document-level, section-aware.

    Pages are cleaned (`clean.strip_page_furniture`) and concatenated into one
    document with a char→page map, so a chunk gets every page its text spans.
    Figure/table number-soup windows are dropped (`clean.is_soup`).
    Bibliography is *not* dropped here — see the module docstring. `chunk_id`
    stays `{paper}::p{first_page}::c{counter}` with `counter` global per paper
    — ids are *expected* to differ from the pre-ADR-0017 corpus; the golden
    set was re-anchored against this output.
    """
    if not pages:
        return []

    paper_id = pages[0].paper_id
    page_texts = [p.text for p in pages]
    header = detect_running_header(page_texts)

    doc_parts: list[str] = []
    bounds: list[tuple[int, int, int]] = []  # (page_number, start, end) in doc
    cursor = 0
    for page, raw in zip(pages, page_texts, strict=True):
        cleaned = strip_page_furniture(raw, header)
        doc_parts.append(cleaned)
        bounds.append((page.page_number, cursor, cursor + len(cleaned)))
        cursor += len(cleaned) + 1  # +1 for the "\n" join separator
    doc = "\n".join(doc_parts)

    chunks: list[Chunk] = []
    counter = 0
    for sec_start, sec_end, title in _section_spans(doc):
        sec_text = doc[sec_start:sec_end]
        for win_start, win_end in _window_spans(sec_text, target_chars, overlap_chars):
            a = sec_start + win_start
            b = sec_start + win_end
            text = sec_text[win_start:win_end].strip()
            if not text or is_soup(text):
                continue
            page_numbers = sorted({pn for pn, s, e in bounds if s < b and e > a})
            if not page_numbers:
                page_numbers = [pages[0].page_number]
            chunks.append(
                Chunk(
                    chunk_id=f"{paper_id}::p{page_numbers[0]}::c{counter}",
                    paper_id=paper_id,
                    page_numbers=page_numbers,
                    text=text,
                    section=title,
                )
            )
            counter += 1
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
