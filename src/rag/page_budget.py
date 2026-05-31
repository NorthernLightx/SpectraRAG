"""Route-by-fit page selector for the production /answer path (ADR 0024).

ADR 0024 measured that on documents that fit the model's context, feeding the
WHOLE document's page images beats a top-k RAG cut (+0.12 where it fits, on
MMLongBench). The eval harness already implements this
(`scripts/experiments/run_mmlb_qa.py:route_pages_by_fit`, behind `--page-budget`).
This module is the production counterpart, deliberately scoped to PAPER-SCOPED
queries only — i.e. when the caller has already named the document via
`Query.filters['paper_id']` (ADR 0009). That is the exact single-doc regime the
win was measured in; corpus-wide document identification stays out of scope
(ADR 0024 §"What this leaves open" — feeding "the whole document" has no
referent until you know which document).

The whole-doc page count is the load-bearing input and MUST come from disk (the
rendered-pages directory), not from retrieval: retrieval surfaces only top-k, so
deriving "all pages" from the retrieved set silently under-feeds the model and
erases the win. `resolve_whole_doc_pages` returns None on any unresolvable doc or
over-budget doc, so the caller falls back to RAG loudly rather than feeding a
truncated document.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.types import RetrievalResult

# Page render filename, mirroring the layout ingestion writes and
# Generator._collect_image_paths reads: `<pages_dir>/<paper>/<paper>_p<N>.png`.
_PAGE_FILE_RE = re.compile(r"^(?P<paper>.+)_p(?P<page>\d+)\.png$")

# `paper_id` arrives from untrusted request input (Query.filters['paper_id']) and
# is used as a path component, so it must be validated before touching the
# filesystem or a value like ".." / "../../etc" would let the whole-doc path
# enumerate and feed page images from outside the corpus. Real ids are arXiv
# (`2310.05634v2`), hashes, or slug names (`05-03-18-political-release`) — all
# within this class. The character class already excludes `/` and `\`; the
# explicit `..` / leading-dot checks close the gap that `[\w.-]+` leaves open
# (it would match "..").
_SAFE_PAPER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _is_safe_paper_id(paper_id: str) -> bool:
    return bool(_SAFE_PAPER_ID_RE.fullmatch(paper_id)) and ".." not in paper_id


def _doc_page_numbers(paper_id: str, pages_dir: Path) -> list[int]:
    """Sorted page numbers for `paper_id` from the rendered-pages directory.

    Empty when the paper directory is absent or holds no matching renders. The
    filename's paper segment must equal `paper_id` exactly so a prefix collision
    (e.g. `2310.05` vs `2310.05634`) can't pull another doc's pages.
    """
    paper_dir = pages_dir / paper_id
    if not paper_dir.is_dir():
        return []
    pages: list[int] = []
    for png in paper_dir.glob(f"{paper_id}_p*.png"):
        m = _PAGE_FILE_RE.match(png.name)
        if m is not None and m.group("paper") == paper_id:
            pages.append(int(m.group("page")))
    return sorted(pages)


def resolve_whole_doc_pages(
    paper_id: str, pages_dir: Path, budget: int
) -> list[RetrievalResult] | None:
    """Whole-document page images as visual RetrievalResults when the doc fits the
    budget, else None (caller falls back to top-k RAG).

    The fit test is the closed interval `page_count <= budget` (ADR 0024). Returns
    None when the doc can't be resolved (missing directory, no rendered pages) or
    exceeds the budget, so the caller never silently feeds a partial document.

    Each page becomes a visual RetrievalResult whose chunk_id mirrors
    `visual.py:_PAGE_CHUNK_FMT` (`<paper>::p<N>::page`, source "visual"), so the
    Generator's existing image-attachment and citation logic treat these
    identically to real visual retrievals. `text` is empty: the page image is the
    payload, and a vision model reads it directly. `score` is 1.0 — the operator
    explicitly scoped the query to this document, so the whole-doc feed must not
    trip the generator's low-confidence refusal gate.

    Security: `paper_id` is untrusted request input used as a path component.
    Reject anything outside the safe-id allowlist, and verify the resolved paper
    directory stays inside `pages_dir` (defeats symlink escapes a pure-string
    check would miss). Any rejection returns None, so the route falls back to RAG.
    """
    if not _is_safe_paper_id(paper_id):
        return None
    base = pages_dir.resolve()
    resolved = (pages_dir / paper_id).resolve()
    if not resolved.is_relative_to(base):
        return None
    pages = _doc_page_numbers(paper_id, pages_dir)
    if not pages or len(pages) > budget:
        return None
    return [
        RetrievalResult(
            chunk_id=f"{paper_id}::p{n}::page",
            paper_id=paper_id,
            score=1.0,
            text="",
            page_numbers=[n],
            source="visual",
        )
        for n in pages
    ]
