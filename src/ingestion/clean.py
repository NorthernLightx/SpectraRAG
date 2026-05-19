"""Page-text hygiene applied before chunking (ADR 0017).

Evidence: ~1 in 5 chunks in the raw corpus is noise the retriever and the
GraphRAG entity-extractor must never see — repeated running-page headers,
bare page-number lines, and figure/table interiors that leak into the PDF
text layer as digit soup. PyMuPDF emits the header and page number as their
*own lines* at the top of each page's text, which makes line-based stripping
exact. Soup detection is a content predicate applied per window after
chunking. Bibliography removal is structural and lives in `chunking.py`
(it needs section headings), not here.
"""

from __future__ import annotations

import re
from collections import Counter

_PAGENO_LINE = re.compile(r"^\d{1,4}$")
_LONG_WORD = re.compile(r"[A-Za-z]{3,}")
# Real text always has these; a page of axis ticks / a value grid does not.
_PROSE_HINT = re.compile(r"\b(the|and|that|with|this|from|which|where|are)\b", re.IGNORECASE)


def _first_nonempty_line(text: str) -> str | None:
    for line in text.splitlines():
        s = " ".join(line.split())
        if s:
            return s
    return None


def detect_running_header(page_texts: list[str], *, min_share: float = 0.6) -> str | None:
    """Most common first non-empty line, if it repeats on >= `min_share` of pages.

    Academic PDFs print a fixed header ("Preprint. Under review.", a journal
    line) on most pages; PyMuPDF puts it on its own line at the top of every
    page's text. Page 1 (title page) often differs, so this is a share
    threshold, not "all pages". Returns None when no line dominates — papers
    with no running header (common) must be left untouched.
    """
    firsts = [ln for t in page_texts if (ln := _first_nonempty_line(t)) is not None]
    if len(firsts) < 3:
        return None
    line, count = Counter(firsts).most_common(1)[0]
    if count >= max(2, int(min_share * len(firsts))) and 3 <= len(line) <= 90:
        return line
    return None


def strip_page_furniture(text: str, header: str | None) -> str:
    """Drop the leading running-header / page-number lines and a trailing page no.

    Only the first few lines are inspected for furniture so a legitimate
    numeric line deeper in the page (e.g. a real "2048" in body text) is never
    removed. A bare number as the last line is a footer page number.
    """
    lines = text.splitlines()
    i, dropped = 0, 0
    while i < len(lines) and dropped < 3:
        s = " ".join(lines[i].split())
        if s == "":
            i += 1
            continue
        if (header is not None and s == header) or _PAGENO_LINE.match(s):
            i += 1
            dropped += 1
            continue
        break
    j = len(lines)
    while j > i:
        s = " ".join(lines[j - 1].split())
        if s == "":
            j -= 1
            continue
        if _PAGENO_LINE.match(s):
            j -= 1
            break
        break
    return "\n".join(lines[i:j])


def is_soup(text: str) -> bool:
    """True when a chunk is a figure/table interior (axis ticks, a value grid).

    These leak from vector-drawn figures into the PDF text layer and are pure
    retrieval noise — no entity or relation is extractable from
    "2.127 2.126 2.134". Heuristic: very low alphabetic density AND few real
    words AND no prose function words. The prose-hint guard keeps
    equation-dense but meaningful passages (which still contain "the/where/…")
    from being dropped; the eval's equation subset is the backstop.
    """
    body = text.strip()
    non_space = [c for c in body if not c.isspace()]
    if not non_space:
        return True
    alpha_ratio = sum(c.isalpha() for c in non_space) / len(non_space)
    long_words = len(_LONG_WORD.findall(body))
    if _PROSE_HINT.search(body) and long_words >= 8:
        return False
    return alpha_ratio < 0.45 and long_words < 12
