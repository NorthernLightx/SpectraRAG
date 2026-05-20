"""ADR 0020 Docling probe — deterministic fast-path candidate.

Same kill-spike discipline as ADR 0018/0020: does a heavyweight ML
document parser (Docling) recover the audit-flagged miss class on
2604.22753v1 (Figures 2/3, Tables 1/3/4) *deterministically* —
without needing the VLM fallback at all on the easy 86 % of pages?

Continue / kill:
- recovers ≥4 of 5 misses + matches Figure 1 (the control) on the right
  page → continue, Docling becomes the fast path, VLM is the residual;
- recovers ≤2 / 5, or invents many false positives → kill, stay with
  PyMuPDF + VLM cascade per ADR 0020 as originally drafted.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from docling.document_converter import DocumentConverter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDF = Path("data/papers/2604.22753v1.pdf")

# 2604.22753v1 ground truth from the overlay audit + the chunk dump.
EXPECTED_FIGURES = {1, 2, 3}
EXPECTED_TABLES = {1, 2, 3, 4}


def main() -> None:
    print(f"converting {PDF} ...")
    started = time.monotonic()
    converter = DocumentConverter()
    result = converter.convert(PDF)
    doc = result.document
    elapsed = time.monotonic() - started
    print(f"converted in {elapsed:.1f}s\n")

    pictures = list(getattr(doc, "pictures", []))
    tables = list(getattr(doc, "tables", []))
    print(f"pictures: {len(pictures)}   tables: {len(tables)}\n")

    fig_pages: dict[int, list[str]] = {}
    tab_pages: dict[int, list[str]] = {}
    for i, pic in enumerate(pictures, start=1):
        provs = getattr(pic, "prov", []) or []
        for prov in provs:
            page = getattr(prov, "page_no", None)
            bbox = getattr(prov, "bbox", None)
            cap_obj = pic.caption_text(doc) if hasattr(pic, "caption_text") else ""
            cap = str(cap_obj or "")[:80]
            fig_pages.setdefault(page or 0, []).append(f"#{i} bbox={bbox} caption: {cap}")
    for i, tab in enumerate(tables, start=1):
        provs = getattr(tab, "prov", []) or []
        for prov in provs:
            page = getattr(prov, "page_no", None)
            bbox = getattr(prov, "bbox", None)
            cap_obj = tab.caption_text(doc) if hasattr(tab, "caption_text") else ""
            cap = str(cap_obj or "")[:80]
            tab_pages.setdefault(page or 0, []).append(f"#{i} bbox={bbox} caption: {cap}")

    all_pages = sorted(set(fig_pages) | set(tab_pages))
    for pg in all_pages:
        print(f"--- p{pg:02d} ---")
        for line in fig_pages.get(pg, []):
            print(f"  FIG {line}")
        for line in tab_pages.get(pg, []):
            print(f"  TAB {line}")
        print()

    # Sniff caption labels (e.g., "Figure 2:") from captions to compare to
    # the known ground truth labels.
    import re

    def labels(by_page: dict[int, list[str]], kind: str) -> set[int]:
        pattern = re.compile(rf"{kind}\s+(\d+)", re.IGNORECASE)
        found: set[int] = set()
        for entries in by_page.values():
            for line in entries:
                m = pattern.search(line)
                if m:
                    try:
                        found.add(int(m.group(1)))
                    except ValueError:
                        pass
        return found

    seen_figs = labels(fig_pages, "Figure")
    seen_tabs = labels(tab_pages, "Table")
    print(f"figure labels seen via captions:  {sorted(seen_figs)}")
    print(f"table  labels seen via captions:  {sorted(seen_tabs)}")
    print(f"expected figures:                 {sorted(EXPECTED_FIGURES)}")
    print(f"expected tables:                  {sorted(EXPECTED_TABLES)}")
    print()
    print(
        f"figures recovered: {len(seen_figs & EXPECTED_FIGURES)} / {len(EXPECTED_FIGURES)}  "
        f"(missing: {sorted(EXPECTED_FIGURES - seen_figs)})"
    )
    print(
        f"tables  recovered: {len(seen_tabs & EXPECTED_TABLES)} / {len(EXPECTED_TABLES)}  "
        f"(missing: {sorted(EXPECTED_TABLES - seen_tabs)})"
    )
    extra_figs = seen_figs - EXPECTED_FIGURES
    extra_tabs = seen_tabs - EXPECTED_TABLES
    if extra_figs:
        print(
            f"unexpected figure labels: {sorted(extra_figs)}  (could be appendix figures, eyeball)"
        )
    if extra_tabs:
        print(
            f"unexpected table labels:  {sorted(extra_tabs)}  (could be appendix tables, eyeball)"
        )


if __name__ == "__main__":
    main()
