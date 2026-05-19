"""One-off: size the junk in the current corpus (backs ADR 0017 before-numbers).

Runs the *actual* current pipeline (extract_pages -> chunk_pages) over every
PDF in data/papers/ and classifies each chunk as: references/bibliography,
numeric/symbol soup, running-header-led, or content. Heuristics here are for
*measurement only* — the production filters are designed from what this prints.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

from src.ingestion.chunking import chunk_pages
from src.ingestion.pdf import extract_pages
from src.types import Chunk

sys.stdout.reconfigure(encoding="utf-8")  # corpus is full of − ± Σ etc.

_REF_START = re.compile(r"^\s*(references|bibliography)\b", re.IGNORECASE)
_APPENDIX_START = re.compile(r"^\s*(appendix\b|[A-Z]\s+[A-Z][a-z])")
_LONG_WORD = re.compile(r"[A-Za-z]{3,}")


def _strip_header(text: str, header: str | None) -> str:
    if header and text.lstrip().lower().startswith(header.lower()):
        return text.lstrip()[len(header) :].lstrip()
    return text


def _running_header(pages_text: list[str]) -> tuple[str | None, int]:
    """Longest leading word-sequence shared by >=60% of pages (the running header).

    Grows the prefix word by word while a majority of pages still share it, so a
    fixed header ("Preprint. Under review.") is found even though page-specific
    text follows it. Page-1 usually lacks the header, hence the 60% (not 100%).
    """
    norm = [" ".join(t.split()) for t in pages_text if t.strip()]
    if len(norm) < 2:
        return None, 0
    words = [t.split(" ") for t in norm]
    quorum = max(2, int(0.6 * len(norm)))
    best: str | None = None
    best_n = 0
    for k in range(1, 13):
        prefixes: Counter[str] = Counter(
            " ".join(w[:k]) for w in words if len(w) >= k
        )
        if not prefixes:
            break
        pref, n = prefixes.most_common(1)[0]
        if n >= quorum and len(pref) <= 80:
            best, best_n = pref, n
        else:
            break
    return (best, best_n) if best else (None, 0)


def _is_soup(text: str) -> bool:
    body = text.strip()
    non_space = [c for c in body if not c.isspace()]
    if not non_space:
        return True
    alpha = sum(c.isalpha() for c in non_space)
    alpha_ratio = alpha / len(non_space)
    long_words = len(_LONG_WORD.findall(body))
    return alpha_ratio < 0.45 and long_words < 12


def _ref_span(chunks: list[Chunk], header: str | None) -> tuple[int, int]:
    """[start, end) chunk-index span of the references/bibliography block."""
    n = len(chunks)
    start = -1
    for i in range(n // 2, n):  # references live in the back half
        if _REF_START.match(_strip_header(chunks[i].text, header)):
            start = i
            break
    if start < 0:
        return (n, n)
    end = n
    for j in range(start + 1, n):
        if _APPENDIX_START.match(_strip_header(chunks[j].text, header)):
            end = j
            break
    return (start, end)


def main() -> None:
    papers = sorted(Path("data/papers").glob("*.pdf"))
    print(f"{'paper':<16} {'chunks':>6} {'refs':>5} {'soup':>5} {'hdr/pg':>7}  header")
    print("-" * 92)
    tot_chunks = tot_refs = tot_soup = 0
    soup_examples: list[str] = []
    ref_starts: list[str] = []
    for pdf in papers:
        pid = pdf.stem
        pages = extract_pages(pid, pdf)
        chunks = chunk_pages(pages)
        header, hdr_pages = _running_header([p.text for p in pages])
        r0, r1 = _ref_span(chunks, header)
        n_ref = r1 - r0
        ref_idx = set(range(r0, r1))
        if n_ref:
            snip = " ".join(_strip_header(chunks[r0].text, header).split())[:90]
            ref_starts.append(f"  {pid} r0={r0} r1={r1}: {snip}")
        else:
            ref_starts.append(f"  {pid}: NO references block detected")
        n_soup = sum(1 for i, c in enumerate(chunks) if i not in ref_idx and _is_soup(c.text))
        for i, c in enumerate(chunks):
            if i not in ref_idx and _is_soup(c.text) and len(soup_examples) < 8:
                soup_examples.append(f"  [{c.chunk_id}] {' '.join(c.text.split())[:110]}")
        tot_chunks += len(chunks)
        tot_refs += n_ref
        tot_soup += n_soup
        hdr_disp = f"{hdr_pages}/{len(pages)}" if header else "-"
        print(
            f"{pid:<16} {len(chunks):>6} {n_ref:>5} {n_soup:>5} {hdr_disp:>7}  "
            f"{(header or '')[:34]}"
        )
    print("-" * 92)
    junk = tot_refs + tot_soup
    print(
        f"{'TOTAL':<16} {tot_chunks:>6} {tot_refs:>5} {tot_soup:>5}"
        f"   junk={junk} ({100 * junk / tot_chunks:.1f}% of corpus)"
    )
    print("\nreferences-start verification (eyeball these):")
    print("\n".join(ref_starts))
    print("\nsample soup chunks (would be dropped):")
    print("\n".join(soup_examples))


if __name__ == "__main__":
    main()
