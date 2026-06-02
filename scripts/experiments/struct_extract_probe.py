"""Structured-extraction probe (frontier-scout Bet 0, 2026-06-01): does feeding
offline-extracted table/chart text ALONGSIDE the page image beat reading the page
image alone? SIMPLOT/TALENT-style augment-don't-replace, on our oracle subset.

WHY THIS DESIGN (and why it's decidable where prior levers weren't): the benchmark
gold is ~25-30% noisy on the failure slice (format artifacts + wrong labels), so an
ABSOLUTE accuracy number is unreadable. This probe is PAIRED: identical oracle
queries, identical QA reader, the ONLY difference is whether a structured-text block
(tables/charts transcribed offline) is prepended to the prompt. Bad-gold and
format-artifact cards score 0 in BOTH arms -> common-mode noise cancels -> the
per-query delta is clean even though the absolute numbers are not.

  baseline arm : gold page image(s) + question            -> answer   (the existing oracle path)
  struct arm   : structured text (offline extract) + gold page image(s) + question -> answer

The offline extraction runs ONCE per page with a STRONG free model (qwen3-vl:235b
via Ollama :cloud) using a structure-first prompt ("transcribe every table/chart as
TSV") — this is the "move the hard perception offline where you can spend more and
verify" thesis. The QA reader stays the cheap free gemma-4-31b so the A/B isolates
the structured-text contribution, not a model swap. Both answers go through the
official MMLongBench scorer (score_mmlb_qa) — same ruler as every other QA number.

NOT authoring ground truth: gold is the human MMLongBench label; this only changes
what evidence the reader sees. The extracted TSV is model-produced INPUT, clearly an
aid, never written into gold.

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.struct_extract_probe \
        --oracle-run data/eval/runs/exp_mmlb_gen_free_oracle.json \
        --extract-model qwen3-vl:235b-cloud \
        --reader-model google/gemma-4-31b-it:free \
        --out data/eval/runs/struct_extract_probe.json --limit 0
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

from scripts.experiments._openrouter_client import build_openrouter_client
from scripts.experiments.fair_judge import _judge_one
from scripts.experiments.run_mmlb_qa import _chat_vision_openrouter
from scripts.experiments.score_mmlb_qa import (
    _EXTRACT_FAILED,
    _FAIL,
    _eval_score,
    _extract_one,
)
from src.llm.ollama_chat import OllamaChatClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_ROOT = Path(".")
_PAGES = _ROOT / "data/pages"
_PAGE_RE = re.compile(r"_p(\d+)\.png$")

# Structure-first extraction prompt: transcribe, do not answer. The point is a
# careful single pass over the page's structured content, cached and reused.
_EXTRACT_PROMPT = (
    "Transcribe ALL structured visual content on this page as plain text:\n"
    "- Tables: tab-separated, header row then data rows.\n"
    "- Charts/plots: each series and its data points (label: value), plus axis labels.\n"
    "- Maps / colour-coded figures: the legend — what each colour or pattern represents "
    "(e.g. 'red = 0-375 miles'), and any labelled regions.\n"
    "- Diagrams: each labelled box/node and its colour.\n"
    "Do NOT answer any question — only transcribe what is visibly present. If the page "
    "has no table, chart, map, or labelled diagram, reply 'NONE'."
)

_BASE_SYS = (
    "You are a careful research assistant answering a question from the page image(s) "
    "of a document. Use only what is visible. Answer directly and concisely; if the "
    'images do not contain the answer, reply exactly "Not answerable".'
)
_STRUCT_SYS = (
    "You are a careful research assistant. You are given (1) STRUCTURED TEXT "
    "transcribed from the page's tables/charts, and (2) the page image(s). Use both; "
    "prefer the structured text for exact numbers. Answer directly and concisely; if "
    'the answer is not present, reply exactly "Not answerable".'
)
_BASE_USER = "Question: {q}\n\nAnswer using the {n} page image(s)."
_STRUCT_USER = "STRUCTURED TEXT FROM PAGE(S):\n{s}\n\nQuestion: {q}\n\nAnswer using the structured text above and the {n} page image(s)."


def _pages_for(paper_id: str, pages: list[int]) -> list[Path]:
    out = []
    for p in pages:
        fp = _PAGES / paper_id / f"{paper_id}_p{p}.png"
        if fp.exists():
            out.append(fp)
    return out


async def _extract_page(client: httpx.AsyncClient, url: str, model: str, img: Path) -> str:
    """One offline structure-first transcription of a page (Ollama /api/chat, base64)."""
    b64 = base64.standard_b64encode(img.read_bytes()).decode()
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": _EXTRACT_PROMPT, "images": [b64]}],
        "options": {"temperature": 0, "num_predict": 700},
    }
    for attempt in range(1, 5):
        try:
            r = await client.post(f"{url}/api/chat", json=payload)
            r.raise_for_status()
            return (r.json().get("message", {}).get("content", "") or "").strip()
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            if attempt == 4:
                return f"__extract_failed__: {type(exc).__name__}"
            await asyncio.sleep(min(2.0 * attempt, 12.0))
    return "__extract_failed__"


async def _ask(
    reader: Any, args: argparse.Namespace, cache: dict[str, Any], qid: str, arm: str,
    sys_p: str, user_p: str, imgs: list[Path]
) -> str:
    """One QA arm, cached. Hoisted to module scope so it binds explicit params, not
    loop variables (ruff B023)."""
    k = f"qa::{arm}::{args.reader_model}::{qid}"
    if k in cache:
        return str(cache[k])
    txt, _, _ = await _chat_vision_openrouter(
        reader, args.reader_model, sys_p, user_p, imgs, temperature=0.0, max_tokens=512
    )
    cache[k] = txt
    args.cache.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return txt


async def _score(
    extractor_text: Any, args: argparse.Namespace, cache: dict[str, Any], qid: str, arm: str,
    q: str, gold: str, fmt: str, ans: str
) -> float:
    """Stage-2 extract + official score for one arm, cached."""
    k = f"sc::{arm}::{qid}"
    pred = cache.get(k)
    if pred is None:
        pred = await _extract_one(extractor_text, args.score_extractor, q, ans)
        if pred != _EXTRACT_FAILED:
            cache[k] = pred
            args.cache.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    pm = "" if pred in (_FAIL, _EXTRACT_FAILED) else pred
    return _eval_score(gold, pm, fmt)


async def _fair(
    reader: Any, args: argparse.Namespace, cache: dict[str, Any], qid: str, arm: str,
    q: str, gold: str, ans: str
) -> bool:
    """Format-tolerant judge (YES/NO) for one arm, cached. True = judged correct."""
    k = f"fair::{arm}::{qid}"
    v = cache.get(k)
    if v is None:
        verdict = await _judge_one(reader, args.fair_model, q, gold, ans)
        v = verdict or "NO"
        cache[k] = v
        args.cache.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return v == "YES"


async def run(args: argparse.Namespace) -> int:
    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    gold_by = {q["query_id"]: q for q in golden["queries"]}
    run = json.loads(args.oracle_run.read_text(encoding="utf-8"))
    per = run["per_query"] if isinstance(run, dict) else run
    if args.restrict and args.restrict.exists():
        raw = json.loads(args.restrict.read_text(encoding="utf-8"))
        keep = {x if isinstance(x, str) else x.get("qid") for x in raw}
        per = [r for r in per if r.get("query_id") in keep]
        print(f"restricted to {len(per)} queries from {args.restrict.name}")
    if args.limit:
        per = per[: args.limit]

    cache: dict[str, Any] = {}
    if args.cache.exists():
        cache = json.loads(args.cache.read_text(encoding="utf-8"))

    reader = build_openrouter_client(timeout=args.timeout)
    extractor_text = OllamaChatClient(base_url=args.ollama_url)  # stage-2 short-answer extractor

    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=args.timeout) as oc:
        for i, rec in enumerate(per):
            qid = rec.get("query_id")
            g = gold_by.get(qid)
            if g is None:
                continue
            gold = str((g.get("expected_facts") or [""])[0])
            fmt = _fmt(g.get("note"))
            q = g.get("text") or rec.get("query") or ""
            paper = g.get("paper_id") or ""
            pages = g.get("relevant_pages") or []
            imgs = _pages_for(paper, pages)
            if not imgs:
                continue

            # --- offline structured extraction (cached per page-set) ---
            ekey = f"extract::{args.extract_model}::{qid}"
            struct = cache.get(ekey)
            if struct is None:
                parts = []
                for img in imgs:
                    parts.append(await _extract_page(oc, args.ollama_url, args.extract_model, img))
                struct = "\n---\n".join(parts)
                cache[ekey] = struct
                args.cache.write_text(json.dumps(cache, indent=2), encoding="utf-8")

            # --- two QA arms (cached) ---
            base_ans = await _ask(
                reader, args, cache, qid, "base", _BASE_SYS, _BASE_USER.format(q=q, n=len(imgs)), imgs
            )
            struct_ans = await _ask(
                reader, args, cache, qid, "struct", _STRUCT_SYS,
                _STRUCT_USER.format(s=struct[:6000], q=q, n=len(imgs)), imgs
            )
            bs = await _score(extractor_text, args, cache, qid, "base", q, gold, fmt, base_ans)
            ss = await _score(extractor_text, args, cache, qid, "struct", q, gold, fmt, struct_ans)
            # Fair (format-tolerant) judge on both arms — the smoke showed strict scoring
            # masks real struct wins as format misses (0002 '0 - 375 miles' vs gold '0-375 miles').
            fb = await _fair(reader, args, cache, qid, "base", q, gold, base_ans)
            fs = await _fair(reader, args, cache, qid, "struct", q, gold, struct_ans)
            flip = "WON" if ss > bs else ("lost" if ss < bs else "same")
            fair_flip = "WON" if fs and not fb else ("lost" if fb and not fs else "same")
            rows.append({"qid": qid, "gold": gold, "fmt": fmt, "category": g.get("category", ""),
                         "base_score": bs, "struct_score": ss, "flip": flip,
                         "fair_base": fb, "fair_struct": fs, "fair_flip": fair_flip,
                         "struct_len": len(struct), "struct_failed": struct.startswith("__extract_failed__")})
            print(f"  [{i+1}/{len(per)}] {qid.split('_')[1]:5} {g.get('category',''):6} "
                  f"strict {bs:.0f}->{ss:.0f} fair {int(fb)}->{int(fs)} [{flip}/{fair_flip}] (ex {len(struct)}c)", flush=True)

    n = len(rows)
    ba = sum(r["base_score"] for r in rows) / n if n else 0.0
    sa = sum(r["struct_score"] for r in rows) / n if n else 0.0
    won = sum(1 for r in rows if r["flip"] == "WON")
    lost = sum(1 for r in rows if r["flip"] == "lost")
    print(f"\nSTRUCT-EXTRACT PROBE  extract={args.extract_model}  read={args.reader_model}  n={n}")
    print(f"  baseline ACC (page only)      = {ba:.4f}")
    print(f"  struct ACC (page + extracted) = {sa:.4f}   (delta {sa - ba:+.4f}; won {won}, lost {lost})")
    print("  (paired: bad-gold/format cards score 0 in BOTH arms, so the delta is noise-robust.)")
    # Fair (format-tolerant) — the smoke showed strict masks struct wins as format misses.
    fba = sum(1 for r in rows if r.get("fair_base")) / n if n else 0.0
    fsa = sum(1 for r in rows if r.get("fair_struct")) / n if n else 0.0
    fwon = sum(1 for r in rows if r.get("fair_flip") == "WON")
    flost = sum(1 for r in rows if r.get("fair_flip") == "lost")
    print(f"\n  FAIR baseline ACC = {fba:.4f}")
    print(f"  FAIR struct ACC   = {fsa:.4f}   (delta {fsa - fba:+.4f}; won {fwon}, lost {flost})")
    # By-category delta: structured extraction should help table/numeric, not colour/spatial.
    from collections import defaultdict
    cats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cats[r["category"]].append(r)
    print("\n  by category (n / base ACC / struct ACC / delta):")
    for cat in sorted(cats):
        cr = cats[cat]
        cb = sum(x["base_score"] for x in cr) / len(cr)
        cs = sum(x["struct_score"] for x in cr) / len(cr)
        print(f"    {cat:14} {len(cr):3}  {cb:.3f} -> {cs:.3f}  ({cs - cb:+.3f})")
    n_none = sum(1 for r in rows if r["struct_len"] <= 6)
    print(f"\n  extraction produced NONE/empty on {n_none}/{n} pages "
          "(structured content absent or not transcribed).")
    args.out.write_text(json.dumps({"extract_model": args.extract_model, "reader_model": args.reader_model,
        "baseline_acc": ba, "struct_acc": sa, "won": won, "lost": lost, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n  Wrote {args.out}")
    return 0


def _fmt(note: str | None) -> str:
    if not note:
        return "Str"
    m = re.search(r"answer_format=(\w+)", note)
    return m.group(1) if m else "Str"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle-run", type=Path, default=Path("data/eval/runs/exp_mmlb_gen_free_oracle.json"))
    ap.add_argument("--golden", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    ap.add_argument("--extract-model", default="qwen3-vl:235b-cloud")
    ap.add_argument("--reader-model", default="google/gemma-4-31b-it:free")
    ap.add_argument("--score-extractor", default="gemma3:4b")
    ap.add_argument("--fair-model", default="openai/gpt-oss-120b:free",
                    help="format-tolerant judge (free OpenRouter); same one fair_judge.py uses")
    ap.add_argument("--restrict", type=Path, default=None,
                    help="JSON list of qids (or objects with 'qid') to restrict to, e.g. the failure set")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--cache", type=Path, default=Path("data/eval/runs/struct_extract_cache.json"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/runs/struct_extract_probe.json"))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
