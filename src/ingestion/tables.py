"""Table extraction from PDFs.

Uses PyMuPDF's built-in `page.find_tables()` (heuristic table detection from
PDF layout). Each detected `Table` is serialised to GitHub-flavoured markdown
and paired with its `Table N:` caption text where one can be located on the
page.

Table detection is imperfect on ArXiv PDFs — multi-page tables, tightly-spaced
columns, and embedded equations all confuse the layout heuristic. We accept
that and let downstream eval tell us where it fails.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from src.observability.logging import get_logger
from src.types import Table

_log = get_logger(__name__)

_TABLE_LABEL_RE = re.compile(
    r"Table\s+(\d+)\s*[:.\-—]\s*([\s\S]*?)"
    r"(?=\n\s*\n|\n\s*(?:Figure|Fig\.?|Table)\s+\d+\s*[:.\-—]|\Z)",
    re.IGNORECASE,
)


def _table_id(paper_id: str, page_no: int, idx: int) -> str:
    return f"{paper_id}::p{page_no}::tab{idx}"


def _extract_captions(page_text: str) -> dict[int, str]:
    """Return `{table_number: caption_body}` parsed from page text."""
    captions: dict[int, str] = {}
    for match in _TABLE_LABEL_RE.finditer(page_text):
        try:
            number = int(match.group(1))
        except ValueError:
            continue
        body = " ".join(match.group(2).split())
        if "\n\n" in body:
            body = body.split("\n\n", 1)[0].strip()
        captions[number] = f"Table {number}: {body}".strip()
    return captions


def _cells_to_markdown(cells: list[list[str | None]]) -> str:
    """Render a 2D list of cells as a GitHub-flavoured markdown table.

    The first row is treated as the header. Empty cells become a single space
    so the markdown still renders. Cells are flattened to a single line each.
    """
    if not cells:
        return ""

    def _norm(cell: str | None) -> str:
        if cell is None:
            return " "
        flat = " ".join(str(cell).split())
        return flat or " "

    width = max(len(row) for row in cells)
    padded = [list(row) + [None] * (width - len(row)) for row in cells]
    header = padded[0]
    body = padded[1:] if len(padded) > 1 else []

    lines = ["| " + " | ".join(_norm(c) for c in header) + " |"]
    lines.append("|" + "|".join(["---"] * width) + "|")
    for row in body:
        lines.append("| " + " | ".join(_norm(c) for c in row) + " |")
    return "\n".join(lines)


def extract_tables(paper_id: str, pdf_path: Path) -> list[Table]:
    """Extract tables from each page of `pdf_path`. Returns one Table per finder hit."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    tables: list[Table] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_no = page.number + 1
            captions = _extract_captions(page.get_text("text") or "")
            try:
                finder = page.find_tables()
            except (RuntimeError, ValueError) as exc:
                _log.warning("table.find_failed", paper_id=paper_id, page=page_no, error=str(exc))
                continue
            for idx, found in enumerate(getattr(finder, "tables", []), start=1):
                try:
                    cells = found.extract()
                except (RuntimeError, ValueError) as exc:
                    _log.warning(
                        "table.extract_failed",
                        paper_id=paper_id,
                        page=page_no,
                        idx=idx,
                        error=str(exc),
                    )
                    continue
                markdown = _cells_to_markdown(cells)
                if not markdown.strip():
                    continue
                tables.append(
                    Table(
                        table_id=_table_id(paper_id, page_no, idx),
                        paper_id=paper_id,
                        page_number=page_no,
                        markdown=markdown,
                        caption=captions.get(idx),
                    )
                )
    return tables
