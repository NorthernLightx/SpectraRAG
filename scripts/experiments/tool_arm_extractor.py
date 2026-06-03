"""Extractor-tool arm: does a STRONGER answer-extractor recover the failures?

The 43 gold-present post-retrieval failures already have a cached baseline answer
(the gemma-4-31b:free reader's free-text answer over the gold page). The official
MMLongBench score runs a stage-2 extractor (default gemma3:4b) to pull a short
answer out of that prose, then matches it. The 2026-05-29 agenda found the
extractor itself is worth ~0.04 (gpt-4o-mini > gemma3:4b), because the weak 4B
extractor drops gold tokens buried in verbose answers.

This arm isolates that lever: hold the reader's baseline ANSWER fixed, swap only
the extractor, re-score. Any lift is pure "better extractor", no new generation,
no perception change — the cleanest possible test of one tool. Sweeps several
free OpenRouter extractors so the result is not one-model-specific.

NOT authoring ground truth: gold answers/formats are the human MMLongBench labels;
this only changes the free-text -> short-answer reduction step.

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.tool_arm_extractor \
        --extractors gemma3:4b z-ai/glm-4.6:free deepseek/deepseek-chat-v3.1:free \
        --out data/eval/runs/tool_arm_extractor.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from scripts.experiments._openrouter_client import build_openrouter_client
from scripts.experiments.score_mmlb_qa import (
    _EXTRACT_FAILED,
    _FAIL,
    _eval_score,
    _extract_one,
)
from src.llm.ollama_chat import OllamaChatClient
from src.llm.protocol import LLMClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_READER = "google/gemma-4-31b-it:free"  # the arm whose baseline answers we re-extract


def _client_for(model: str, ollama_url: str, timeout: float) -> LLMClient:
    # An OpenRouter id has a '/'; a bare name is a local Ollama model.
    if "/" in model:
        return build_openrouter_client(timeout=timeout)
    return OllamaChatClient(base_url=ollama_url)


async def run(args: argparse.Namespace) -> int:
    failures = json.loads(args.failures.read_text(encoding="utf-8"))
    cache = json.loads(args.cache.read_text(encoding="utf-8")) if args.cache.exists() else {}
    ecache: dict[str, str] = {}
    if args.extract_cache and args.extract_cache.exists():
        ecache = json.loads(args.extract_cache.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for f in failures:
        qid = f["qid"]
        ba = cache.get(f"baseline::{_READER}::{qid}")
        if ba is None:
            continue
        rows.append({"qid": qid, "question": f["query"], "gold": str(f["gold"]), "fmt": f["fmt"], "answer": ba})

    print(f"re-extracting {len(rows)} baseline answers across {len(args.extractors)} extractors\n")
    per_extractor: dict[str, dict[str, Any]] = {}
    for model in args.extractors:
        client = _client_for(model, args.ollama_url, args.timeout)
        scores: list[float] = []
        preds: dict[str, str] = {}
        for r in rows:
            key = f"{model}::{r['qid']}"
            pred = ecache.get(key)
            if pred is None:
                pred = await _extract_one(client, model, r["question"], r["answer"])
                if pred != _EXTRACT_FAILED:
                    ecache[key] = pred
                    if args.extract_cache:
                        args.extract_cache.write_text(json.dumps(ecache, indent=2), encoding="utf-8")
            pred_match = "" if pred in (_FAIL, _EXTRACT_FAILED) else pred
            s = _eval_score(r["gold"], pred_match, r["fmt"])
            scores.append(s)
            preds[r["qid"]] = pred
        acc = sum(scores) / len(scores) if scores else 0.0
        wins = [r["qid"] for r, s in zip(rows, scores, strict=True) if s > 0.5]
        per_extractor[model] = {"acc": acc, "n": len(scores), "wins": wins, "preds": preds}
        print(f"  {model:42} ACC={acc:.4f}  recovered={len(wins)}/{len(scores)}  {[w.split('_')[1] for w in wins]}")

    base_model = args.extractors[0]
    base_wins = set(per_extractor[base_model]["wins"])
    print(f"\nIncremental recovery vs baseline extractor ({base_model}):")
    for model in args.extractors[1:]:
        w = set(per_extractor[model]["wins"])
        gained = w - base_wins
        lost = base_wins - w
        print(f"  {model:42} +{len(gained)} -{len(lost)}  net {len(w) - len(base_wins):+d}  gained={[g.split('_')[1] for g in gained]}")

    args.out.write_text(json.dumps({"reader": _READER, "extractors": args.extractors, "per_extractor": per_extractor}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--failures", type=Path, default=Path("docs/research/2026-05-29-agenda/postret_failures.json"))
    ap.add_argument("--cache", type=Path, default=Path("data/eval/runs/crop_reread_cache.json"))
    ap.add_argument("--extract-cache", type=Path, default=Path("data/eval/runs/tool_arm_extract_cache.json"))
    ap.add_argument("--extractors", nargs="+", default=["gemma3:4b"], help="first is the baseline extractor; rest are challengers")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", type=Path, default=Path("data/eval/runs/tool_arm_extractor.json"))
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
