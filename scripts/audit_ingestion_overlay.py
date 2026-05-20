"""Per-page visual audit of figure/table ingestion (ADR 0017/0011 follow-up).

The structural scorecard (`scripts.eval_ingestion`) counts chunks; it cannot
tell you whether *this* figure on *this* page actually got extracted with a
sensible bbox. This tool does:

- Renders each page, draws orange boxes for extracted figures (figure_id +
  caption snippet) and blue boxes for extracted tables (table_id + caption).
- Scans every page's text for `Figure N:` / `Table N:` captions and reports
  the labels that the *captions say exist* per page — then flags pages
  where the extractor produced nothing despite a caption mention. That is
  the most likely silent-miss class.

Output: `data/eval/ingestion/overlays/<paper_id>/p{NN}.png` per page plus
`<paper_id>/audit.md` with the per-paper miss table. No source files are
modified; bboxes are drawn on an in-memory copy of the page.

    uv run python -m scripts.audit_ingestion_overlay --paper 2604.22753v1
    uv run python -m scripts.audit_ingestion_overlay --all
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz

from src.ingestion.figures import extract_figures
from src.ingestion.tables import extract_tables
from src.observability.logging import get_logger
from src.types import Figure, Table

_log = get_logger(__name__)

# Same caption-label regexes the extractors use (kept local so this tool is
# a *check*, not just a re-call of the extraction code's own self-claim).
_FIG_LABEL_RE = re.compile(r"(?:Figure|Fig\.?)\s+((?:[A-Z]\.?)?\d+(?:\.\d+)*)\s*[:.\-—]")
_TAB_LABEL_RE = re.compile(r"Table\s+(\d+)\s*[:.\-—]")

# Render DPI matches `src/ingestion/visual.py` so bbox→pixel scaling is exact.
_DPI = 150
_SCALE = _DPI / 72.0

# Colors are (r, g, b) in 0-1 (PyMuPDF convention).
_FIG_COLOR = (1.0, 0.5, 0.0)  # orange
_TAB_COLOR = (0.0, 0.4, 1.0)  # blue


def _labels_in_text(text: str) -> tuple[set[str], set[str]]:
    """`(figure_labels, table_labels)` extracted from a page's raw text.

    Returns the labels (e.g., `"1"`, `"E.1"`, `"3"`) the captions on this
    page *claim* exist, so the audit can compare against what
    `extract_figures` / `extract_tables` actually produced.
    """
    figs = {m.group(1).strip() for m in _FIG_LABEL_RE.finditer(text) if m.group(1)}
    tabs = {m.group(1).strip() for m in _TAB_LABEL_RE.finditer(text) if m.group(1)}
    return figs, tabs


def _figure_label(fig: Figure) -> str | None:
    """Pull the `N` out of `"Figure N: ..."` for matching against page text."""
    if not fig.caption:
        return None
    m = _FIG_LABEL_RE.search(fig.caption)
    return m.group(1).strip() if m else None


def _table_label(tab: Table) -> str | None:
    if not tab.caption:
        return None
    m = _TAB_LABEL_RE.search(tab.caption)
    return m.group(1).strip() if m else None


def _draw_box(page: fitz.Page, bbox: tuple[float, float, float, float], color: tuple[float, float, float], label: str) -> None:
    rect = fitz.Rect(*bbox)
    page.draw_rect(rect, color=color, width=1.5)
    # Tiny label tag in the top-left corner of the box.
    tag_rect = fitz.Rect(rect.x0, rect.y0 - 12, rect.x0 + 6 * len(label) + 6, rect.y0)
    page.draw_rect(tag_rect, color=color, fill=color, width=0)
    page.insert_text(
        (rect.x0 + 3, rect.y0 - 3),
        label,
        fontname="helv",
        fontsize=7,
        color=(1, 1, 1),
    )


def _audit_paper(paper_id: str, out_root: Path) -> dict[str, object]:
    """Render annotated pages + return per-paper audit data."""
    pdf_path = Path(f"data/papers/{paper_id}.pdf")
    figures = extract_figures(paper_id, pdf_path, out_dir=Path("data/figures"))
    tables = extract_tables(paper_id, pdf_path)

    figs_by_page: dict[int, list[Figure]] = {}
    for f in figures:
        figs_by_page.setdefault(f.page_number, []).append(f)
    tabs_by_page: dict[int, list[Table]] = {}
    for t in tables:
        tabs_by_page.setdefault(t.page_number, []).append(t)

    out_dir = out_root / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)
    misses: list[str] = []
    pages_total = 0
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_no = page.number + 1
            pages_total += 1
            page_text = page.get_text("text") or ""
            caption_figs, caption_tabs = _labels_in_text(page_text)
            extracted_figs = figs_by_page.get(page_no, [])
            extracted_tabs = tabs_by_page.get(page_no, [])

            # Overlay extracted artifacts.
            for f in extracted_figs:
                if f.bbox is None:
                    continue
                label = f"F{_figure_label(f) or '?'} {f.figure_id.rsplit('::', 1)[-1]}"
                _draw_box(
                    page,
                    (f.bbox.x0, f.bbox.y0, f.bbox.x1, f.bbox.y1),
                    _FIG_COLOR,
                    label[:24],
                )
            for t in extracted_tabs:
                if t.bbox is None:
                    continue
                label = f"T{_table_label(t) or '?'} {t.table_id.rsplit('::', 1)[-1]}"
                _draw_box(
                    page,
                    (t.bbox.x0, t.bbox.y0, t.bbox.x1, t.bbox.y1),
                    _TAB_COLOR,
                    label[:24],
                )

            pix = page.get_pixmap(matrix=fitz.Matrix(_SCALE, _SCALE), alpha=False)
            pix.save(str(out_dir / f"p{page_no:02d}.png"))

            # Miss detection: caption says label exists, extractor produced no
            # artifact with that label on this page. Bboxes-less extracted
            # artifacts also count as visually-incomplete misses.
            extracted_fig_labels = {_figure_label(f) for f in extracted_figs if _figure_label(f)}
            extracted_tab_labels = {_table_label(t) for t in extracted_tabs if _table_label(t)}
            missing_figs = sorted(caption_figs - extracted_fig_labels)
            missing_tabs = sorted(caption_tabs - extracted_tab_labels)
            no_bbox_figs = sum(1 for f in extracted_figs if f.bbox is None)
            no_bbox_tabs = sum(1 for t in extracted_tabs if t.bbox is None)
            if missing_figs or missing_tabs or no_bbox_figs or no_bbox_tabs:
                misses.append(
                    f"p{page_no:02d}: "
                    + ", ".join(
                        x for x in [
                            f"missing Figure {missing_figs}" if missing_figs else "",
                            f"missing Table {missing_tabs}" if missing_tabs else "",
                            f"{no_bbox_figs} figure(s) without bbox" if no_bbox_figs else "",
                            f"{no_bbox_tabs} table(s) without bbox" if no_bbox_tabs else "",
                        ] if x
                    )
                )

    miss_lines = (
        [f"- {m}" for m in misses]
        if misses
        else ["- _(none — every captioned label was matched by an extracted artifact with a bbox)_"]
    )
    audit_md = [
        f"# Ingestion overlay audit — `{paper_id}`",
        "",
        f"- pages: **{pages_total}**",
        f"- figures extracted: **{len(figures)}**  (with bbox: "
        f"{sum(1 for f in figures if f.bbox is not None)})",
        f"- tables extracted: **{len(tables)}**  (with bbox: "
        f"{sum(1 for t in tables if t.bbox is not None)})",
        f"- pages flagged: **{len(misses)} / {pages_total}**",
        "",
        "## Flagged pages",
        "",
        *miss_lines,
        "",
        "Legend: orange = extracted figure; blue = extracted table. Labels read "
        "`F<caption-num> <chunk-id-tail>` / `T<caption-num> <chunk-id-tail>`.",
    ]
    (out_dir / "audit.md").write_text("\n".join(audit_md), encoding="utf-8")
    return {
        "paper_id": paper_id,
        "pages": pages_total,
        "figures": len(figures),
        "tables": len(tables),
        "figures_with_bbox": sum(1 for f in figures if f.bbox is not None),
        "tables_with_bbox": sum(1 for t in tables if t.bbox is not None),
        "flagged_pages": len(misses),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--paper", help="one paper_id, e.g. 2604.22753v1")
    g.add_argument("--all", action="store_true", help="audit every PDF under data/papers/")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/eval/ingestion/overlays"),
        help="root for per-paper overlay PNGs + audit.md",
    )
    args = ap.parse_args()

    paper_ids = (
        sorted(p.stem for p in Path("data/papers").glob("*.pdf"))
        if args.all
        else [args.paper]
    )
    print(f"auditing {len(paper_ids)} paper(s) -> {args.out_dir}")
    rollup: list[dict[str, object]] = []
    for pid in paper_ids:
        print(f"  {pid} ...", end=" ", flush=True)
        rollup.append(_audit_paper(pid, args.out_dir))
        print("ok")

    print("\nROLLUP")
    print(f"{'paper':<16} {'pages':>5} {'figs':>5} {'with bbox':>9} {'tables':>6} {'with bbox':>9} {'flagged':>7}")
    for r in rollup:
        print(
            f"{r['paper_id']:<16} {r['pages']:>5} {r['figures']:>5} "
            f"{r['figures_with_bbox']:>9} {r['tables']:>6} {r['tables_with_bbox']:>9} {r['flagged_pages']:>7}"
        )


if __name__ == "__main__":
    main()
