"""DPI arm: does rendering the gold page at HIGHER resolution recover misreads?

The committed page renders are 150 DPI (src/ingestion/visual.py:_DEFAULT_DPI), and
some are lower (the DETR doc rendered at ~117 DPI / 863px long edge). If a "misread"
is really "the text/colour was too small to resolve at 150 DPI", then re-rendering
the SAME gold page at a higher DPI and re-asking the SAME reader should recover it —
a pure resolution lever, no model change, no crop, no extra reasoning.

This is the clean control for ADR 0025's "perception-bound" conclusion: that ADR
showed the reader can't LOCALIZE (its self-chosen crop hurt). DPI tests the other
half — whether the reader can't RESOLVE. If high-DPI recovers misreads, the ceiling
is partly a render-quality bug, not the model's weights, and the fix is ingestion
(raise DPI) not a tool.

For each qid: re-render its gold page(s) from the source PDF at --dpi, feed to the
reader with the baseline prompt, score officially. Compare to the cached 150-DPI
baseline answer. Source PDFs live in data/mmlongbench/documents/<paper_id>.pdf.

NOT authoring ground truth: gold is the human MMLongBench label; this only changes
the render resolution of the input image.

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.tool_arm_dpi \
        --qids mmlb_0080_2005.12872v3 mmlb_0076_2005.12872v3 mmlb_0060_2303.05039v2 ... \
        --dpi 300 --out data/eval/runs/tool_arm_dpi.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from scripts.experiments._openrouter_client import build_openrouter_client
from scripts.experiments.run_mmlb_qa import _SYSTEM_PROMPT, _chat_vision_openrouter
from scripts.experiments.score_mmlb_qa import (
    _EXTRACT_FAILED,
    _FAIL,
    _eval_score,
    _extract_one,
)
from src.llm.ollama_chat import OllamaChatClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_READER = "google/gemma-4-31b-it:free"
_PDF_ROOT = Path("data/mmlongbench/documents")
_BASELINE_USER = "Question: {query}\n\nAnswer using only the {n_pages} page image(s) above."
# Gold page filename: <paper>/<paper>_p<N>.png ; paper_id may differ from the
# truncated qid suffix, so we read the true paper + pages from the failure record.
_PAGE_RE = re.compile(r"_p(\d+)\.png$")


def _render(paper_id: str, page_no: int, dpi: int, out_dir: Path) -> Path | None:
    """Render one PDF page to PNG at the given DPI. Returns path or None if the
    source PDF / page is unavailable."""
    pdf = _PDF_ROOT / f"{paper_id}.pdf"
    if not pdf.exists():
        return None
    out = out_dir / f"{paper_id}_p{page_no}_dpi{dpi}.png"
    if out.exists():
        return out
    try:
        with fitz.open(pdf) as doc:
            if page_no - 1 >= len(doc):
                return None
            page = doc[page_no - 1]  # gold pages are 1-based
            zoom = dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            pix.save(out)
        return out
    except Exception:  # a render failure must not abort the sweep
        return None


async def run(args: argparse.Namespace) -> int:
    failures = {f["qid"]: f for f in json.loads(args.failures.read_text(encoding="utf-8"))}
    cache = json.loads(args.cache.read_text(encoding="utf-8")) if args.cache.exists() else {}
    ecache: dict[str, str] = {}
    if args.extract_cache and args.extract_cache.exists():
        ecache = json.loads(args.extract_cache.read_text(encoding="utf-8"))

    reader = build_openrouter_client(timeout=args.timeout)
    extractor = OllamaChatClient(base_url=args.ollama_url)
    out_dir = args.out.parent / f"_dpi{args.dpi}_renders"
    out_dir.mkdir(parents=True, exist_ok=True)

    async def extract_score(qid: str, arm: str, q: str, ans: str, gold: str, fmt: str) -> float:
        key = f"{arm}::{qid}"
        pred = ecache.get(key)
        if pred is None:
            pred = await _extract_one(extractor, args.extractor_model, q, ans)
            if pred != _EXTRACT_FAILED:
                ecache[key] = pred
                if args.extract_cache:
                    args.extract_cache.write_text(json.dumps(ecache, indent=2), encoding="utf-8")
        pm = "" if pred in (_FAIL, _EXTRACT_FAILED) else pred
        return _eval_score(gold, pm, fmt)

    rows: list[dict[str, Any]] = []
    for qid in args.qids:
        f = failures.get(qid)
        if f is None:
            print(f"  {qid}: not in failures, skip")
            continue
        gold, fmt, q = str(f["gold"]), f["fmt"], f["query"]
        paper_id = f.get("paper_id") or ""
        # derive page numbers from the committed png paths
        pages = [int(m.group(1)) for p in f["gold_page_pngs"] if (m := _PAGE_RE.search(p))]
        # paper_id from the png path (the dir name), robust to qid truncation
        if not paper_id and f["gold_page_pngs"]:
            paper_id = Path(f["gold_page_pngs"][0]).parent.name

        rendered = [_render(paper_id, pg, args.dpi, out_dir) for pg in pages]
        rendered = [r for r in rendered if r is not None]
        if not rendered:
            print(f"  {qid.split('_')[1]:6} NO RENDER (pdf/page missing for {paper_id}) -- skip")
            continue

        base_ans = cache.get(f"baseline::{_READER}::{qid}", "")
        base_score = await extract_score(qid, "baseline150", q, base_ans, gold, fmt)

        text, _, _ = await _chat_vision_openrouter(
            reader, _READER, _SYSTEM_PROMPT,
            _BASELINE_USER.format(query=q, n_pages=len(rendered)), rendered,
            temperature=0.0, max_tokens=512,
        )
        hi_score = await extract_score(qid, f"dpi{args.dpi}", q, text, gold, fmt)
        flip = "WON" if hi_score > base_score else ("lost" if hi_score < base_score else "same")
        rows.append({"qid": qid, "gold": gold, "fmt": fmt, "dpi": args.dpi,
            "baseline150_answer": base_ans[:120], "baseline_score": base_score,
            "hidpi_answer": text[:120], "hidpi_score": hi_score, "flip": flip})
        print(f"  {qid.split('_')[1]:6} gold={gold[:16]:16} base150={base_score:.1f} dpi{args.dpi}={hi_score:.1f} [{flip}]  hi={text[:50]!r}")

    n = len(rows)
    bacc = sum(r["baseline_score"] for r in rows) / n if n else 0.0
    hacc = sum(r["hidpi_score"] for r in rows) / n if n else 0.0
    won = sum(1 for r in rows if r["flip"] == "WON")
    lost = sum(1 for r in rows if r["flip"] == "lost")
    print(f"\nDPI ARM  reader={_READER}  dpi={args.dpi}  n={n}")
    print(f"  baseline 150-DPI ACC = {bacc:.4f}")
    print(f"  {args.dpi}-DPI ACC        = {hacc:.4f}  (delta {hacc - bacc:+.4f}; won {won}, lost {lost})")
    print("  (won = higher resolution recovered a misread = it was a RENDER-quality limit, not the model;")
    print("   no-change = misread persists at high DPI = genuinely perception/model-bound)")

    args.out.write_text(json.dumps({"reader": _READER, "dpi": args.dpi, "rows": rows,
        "baseline_acc": bacc, "hidpi_acc": hacc, "won": won, "lost": lost}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--failures", type=Path, default=Path("docs/research/2026-05-29-agenda/postret_failures.json"))
    ap.add_argument("--cache", type=Path, default=Path("data/eval/runs/crop_reread_cache.json"))
    ap.add_argument("--extract-cache", type=Path, default=Path("data/eval/runs/tool_arm_dpi_extract_cache.json"))
    ap.add_argument("--qids", nargs="+", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--extractor-model", default="gemma3:4b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", type=Path, default=Path("data/eval/runs/tool_arm_dpi.json"))
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
