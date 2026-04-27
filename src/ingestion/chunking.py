"""Section-aware chunking. Splits page text on numbered headings, then by char budget."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.types import Chunk, Page

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
