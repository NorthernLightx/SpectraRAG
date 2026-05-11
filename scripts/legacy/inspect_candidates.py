"""One-shot: score candidate PDFs by visual richness for corpus selection.

Reports per-paper page count, figure count (raster images embedded in pages),
table-heuristic count (pages whose text has many short numeric-leaning rows),
title (first non-empty line of page 1), and a single composite score so we can
pick a balanced set of ~15 keepers from a larger candidate pool.

Run:
  uv run python -m scripts.inspect_candidates --candidates data/papers_candidates
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import fitz

_NUMERIC_RE = re.compile(r"^[\s|]*[-+]?\d[\d.,%eE+\-\s|/]*$")


@dataclass(frozen=True)
class Score:
    arxiv_id: str
    pages: int
    figures: int
    pages_with_figures: int
    table_pages: int
    title: str
    composite: float


def _looks_like_table_page(text: str) -> bool:
    """Heuristic: pages with many short numeric rows likely contain a table."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 6:
        return False
    numeric_short = sum(1 for line in lines if 2 <= len(line) <= 60 and _NUMERIC_RE.match(line))
    return numeric_short >= 5


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if len(s) > 5:
            return s
    return ""


def score_pdf(path: Path) -> Score | None:
    try:
        doc = fitz.open(path)
    except Exception as exc:
        print(f"  skip {path.name}: open failed: {exc}")
        return None

    try:
        pages = doc.page_count
        figures = 0
        pages_with_figures = 0
        table_pages = 0
        title_text = ""
        for page_no in range(pages):
            page = doc.load_page(page_no)
            imgs = page.get_images(full=False)
            if imgs:
                figures += len(imgs)
                pages_with_figures += 1
            text = page.get_text("text")
            if _looks_like_table_page(text):
                table_pages += 1
            if page_no == 0:
                title_text = _first_nonempty_line(text)[:120]
    finally:
        doc.close()

    # Composite score weights what we *want* in the corpus for visual eval
    # research: a paper with several figures and at least one table-page beats
    # one with raw page count alone. Hard floor: enough pages that retrieval
    # has somewhere to go but not so many that ColQwen2 GPU embedding blows up.
    pages_ok = 8 <= pages <= 45
    composite = (1.5 * pages_with_figures + 2.0 * table_pages + 0.05 * pages) if pages_ok else 0.0

    return Score(
        arxiv_id=path.stem,
        pages=pages,
        figures=figures,
        pages_with_figures=pages_with_figures,
        table_pages=table_pages,
        title=title_text,
        composite=composite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data/papers_candidates"))
    args = parser.parse_args()

    scores: list[Score] = []
    for pdf in sorted(args.candidates.glob("*.pdf")):
        s = score_pdf(pdf)
        if s is not None:
            scores.append(s)

    scores.sort(key=lambda s: s.composite, reverse=True)
    print(
        f"{'arxiv_id':<14} {'pages':>5} {'figs':>5} {'pgFigs':>6} {'tblP':>4} {'score':>6}  title"
    )
    for s in scores:
        print(
            f"{s.arxiv_id:<14} {s.pages:>5} {s.figures:>5} "
            f"{s.pages_with_figures:>6} {s.table_pages:>4} "
            f"{s.composite:>6.2f}  {s.title}"
        )


if __name__ == "__main__":
    main()
