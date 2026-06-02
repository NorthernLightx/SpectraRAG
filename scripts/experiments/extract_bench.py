"""Structured-extraction benchmark (Goal 2026-06-01): measure extraction recall on
the 29 structured-object cards across the FULL golden, so any extractor can be scored
against the same target set and compared to the 2026 SOTA bars (table TEDS ~88-94).

Distinct from struct_extract_probe.py (which measures QA-lift on the 43 FAILURE cards
— wrong population for extraction). This extracts each structured-object gold page once
and scores whether the gold value is present (recall proxy for TEDS, no external HTML
labels). Pluggable extractor via --backend so qwen-cloud vs a local specialist
(mineru/docling) are measured identically.

NOT authoring gold: target is the human MMLongBench gold value; only checks presence.

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.extract_bench \
        --targets data/eval/audit/_struct_target_qids.json \
        --backend qwen-cloud --out data/eval/runs/extract_bench_qwen.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

from scripts.experiments.extraction_recall import _gold_tokens, _present

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_PAGES = Path("data/pages")
_EXTRACT_PROMPT = (
    "Transcribe ALL structured visual content on this page as plain text:\n"
    "- Tables: tab-separated, header row then data rows, every cell.\n"
    "- Charts/plots: each series and its data points (label: value), axis labels, legend.\n"
    "- Maps/colour figures: the legend (what each colour represents) and labelled regions.\n"
    "Do NOT answer any question — only transcribe. Reply 'NONE' if no structured content."
)


async def _extract_qwen(client: httpx.AsyncClient, url: str, model: str, imgs: list[Path]) -> str:
    out = []
    for img in imgs:
        b64 = base64.standard_b64encode(img.read_bytes()).decode()
        payload = {"model": model, "stream": False,
                   "messages": [{"role": "user", "content": _EXTRACT_PROMPT, "images": [b64]}],
                   "options": {"temperature": 0, "num_predict": 900}}
        for attempt in range(1, 5):
            try:
                r = await client.post(f"{url}/api/chat", json=payload)
                r.raise_for_status()
                out.append((r.json().get("message", {}).get("content", "") or "").strip())
                break
            except (httpx.HTTPStatusError, httpx.TransportError):
                if attempt == 4:
                    out.append("__extract_failed__")
                else:
                    await asyncio.sleep(min(2.0 * attempt, 12.0))
    return "\n---\n".join(out)


def _mineru_extract(imgs: list[Path], mineru_bin: str, out_root: Path, backend: str) -> str:
    """Run MinerU (isolated-venv CLI) on each page image; read back its markdown.
    backend = 'vlm-auto-engine' (MinerU2.5-1.2B local VLM, GPU) or 'pipeline' (OCR)."""
    import os
    import subprocess

    out: list[str] = []
    env = {**os.environ, "MINERU_MODEL_SOURCE": "huggingface"}
    sub = "vlm" if backend.startswith("vlm") else "auto"
    # Windows CreateProcess needs an ABSOLUTE, native-separator path with .exe — a
    # relative forward-slash path that pathlib reports as existing still raises WinError 2.
    bp = Path(mineru_bin)
    if os.name == "nt" and bp.suffix != ".exe" and bp.with_suffix(".exe").exists():
        bp = bp.with_suffix(".exe")
    binpath = str(bp.resolve())
    out_abs = str(out_root.resolve())
    for img in imgs:
        try:
            subprocess.run(
                [binpath, "-p", str(img.resolve()), "-o", out_abs, "-b", backend, "--source", "local"],
                check=True, capture_output=True, timeout=300, env=env,
            )
            md = out_root / img.stem / sub / f"{img.stem}.md"
            out.append(md.read_text(encoding="utf-8") if md.exists() else "__extract_failed__: no md")
        except Exception as exc:  # parser/timeout failure must not abort the bench
            out.append(f"__extract_failed__: {type(exc).__name__}")
    return "\n---\n".join(out)


async def _mineru_server_extract(client: httpx.AsyncClient, url: str, imgs: list[Path]) -> str:
    """Extract via a persistent MinerU API server (model preloaded once = efficient).
    POST each page to /file_parse, read results.<stem>.md_content."""
    out: list[str] = []
    for img in imgs:
        try:
            with img.open("rb") as fh:
                r = await client.post(
                    f"{url}/file_parse",
                    files={"files": (img.name, fh, "image/png")},
                    data={"backend": "vlm-auto-engine", "return_md": "true", "source": "local"},
                    timeout=300,
                )
            r.raise_for_status()
            res = r.json().get("results", {})
            md = next((v["md_content"] for v in res.values()
                       if isinstance(v, dict) and "md_content" in v), "")
            out.append(md or "__extract_failed__: no md_content")
        except Exception as exc:  # one bad page must not abort the bench
            out.append(f"__extract_failed__: {type(exc).__name__}")
    return "\n---\n".join(out)


def _docling_extract(imgs: list[Path]) -> str:
    """Local Docling parse of each page image -> markdown (tables become md tables)."""
    from docling.document_converter import DocumentConverter

    conv = DocumentConverter()
    out = []
    for img in imgs:
        try:
            res = conv.convert(str(img))
            out.append(res.document.export_to_markdown())
        except Exception as exc:  # parser failure must not abort the bench
            out.append(f"__extract_failed__: {type(exc).__name__}")
    return "\n---\n".join(out)


async def run(args: argparse.Namespace) -> int:
    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    gold_by = {q["query_id"]: q for q in golden["queries"]}
    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    cache = json.loads(args.cache.read_text(encoding="utf-8")) if args.cache.exists() else {}

    rows: list[dict[str, Any]] = []
    client = httpx.AsyncClient(timeout=args.timeout)
    try:
        for i, qid in enumerate(targets):
            g = gold_by.get(qid)
            if g is None:
                continue
            paper = g.get("paper_id", "")
            pages = g.get("relevant_pages") or []
            imgs = [_PAGES / paper / f"{paper}_p{p}.png" for p in pages]
            imgs = [p for p in imgs if p.exists()]
            if not imgs:
                continue
            ck = f"{args.backend}::{qid}"
            extract = cache.get(ck)
            if extract is None:
                if args.backend == "qwen-cloud":
                    extract = await _extract_qwen(client, args.ollama_url, args.model, imgs)
                elif args.backend == "docling":
                    extract = await asyncio.to_thread(_docling_extract, imgs)
                elif args.backend == "mineru-vlm":
                    extract = await asyncio.to_thread(
                        _mineru_extract, imgs, args.mineru_bin, args.mineru_out, "vlm-auto-engine"
                    )
                elif args.backend == "mineru-server":
                    extract = await _mineru_server_extract(client, args.mineru_url, imgs)
                else:
                    raise SystemExit(f"unknown backend {args.backend}")
                cache[ck] = extract
                args.cache.write_text(json.dumps(cache, indent=2), encoding="utf-8")
            extract = str(extract)
            gold = str((g.get("expected_facts") or [""])[0])
            toks = _gold_tokens(gold)
            hit = bool(toks) and all(_present(t, extract) for t in toks)
            rows.append({"qid": qid, "category": g.get("category", ""), "gold": gold,
                         "hit": hit, "len": len(extract),
                         "none": extract.strip().upper().startswith("NONE") or len(extract) < 8})
            print(f"  [{i+1}/{len(targets)}] {qid.split('_')[1]:5} {g.get('category',''):6} "
                  f"{'HIT ' if hit else 'miss'} gold={gold[:22]!r} (ex {len(extract)}c)", flush=True)
    finally:
        await client.aclose()

    n = len(rows)
    rec = sum(r["hit"] for r in rows) / n if n else 0.0
    print(f"\nEXTRACT BENCH  backend={args.backend}  n={n} structured-object cards")
    print(f"  recall (gold value present) = {rec:.4f}")
    print("  (proxy for table/chart-to-text quality; 2026 SOTA table TEDS ~88-94)")
    args.out.write_text(json.dumps({"backend": args.backend, "recall": rec, "n": n, "rows": rows},
                                   indent=2), encoding="utf-8")
    print(f"  Wrote {args.out}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=Path, default=Path("data/eval/audit/_struct_target_qids.json"))
    ap.add_argument("--golden", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    ap.add_argument("--backend", choices=("qwen-cloud", "docling", "mineru-vlm", "mineru-server"),
                    default="qwen-cloud")
    ap.add_argument("--mineru-url", default="http://127.0.0.1:8011",
                    help="MinerU API server (mineru-server backend; model preloaded = efficient)")
    ap.add_argument("--mineru-bin", default="tools/_mineru_venv/Scripts/mineru",
                    help="MinerU CLI in the isolated venv (mineru-vlm backend)")
    ap.add_argument("--mineru-out", type=Path, default=Path("tools/_mineru_out"),
                    help="MinerU output dir (mineru-vlm backend)")
    ap.add_argument("--model", default="qwen3-vl:235b-cloud")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--cache", type=Path, default=Path("data/eval/runs/extract_bench_cache.json"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/runs/extract_bench.json"))
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
