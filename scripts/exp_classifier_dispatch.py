"""Measure the LLM classifier's dispatch rate vs ADR 0008's regex on MMLongBench.

The retrieval eval (run cc45831697b6) showed the regex classifier under-fired
by ~75 %: only 26 of 149 queries dispatched to hybrid where 98 were figure/
table-evidenced. The hypothesis was that natural-language questions don't say
"Figure X" / "Table N" so the regex misses them.

This experiment runs both classifiers over MMLongBench's golden v1 (149 queries)
and reports: per-classifier hybrid-dispatch rate, agreement matrix, and the
specific queries where they disagree. If the LLM classifier closes the gap
(dispatches a markedly higher fraction of figure/table-evidenced queries to
hybrid than regex), then composing it with the vision-generator wiring from
commit 82405b4 gives the end-to-end win without a 4-hour live re-run.

Cost: ~149 x $0.0001 ~= $0.02. Wall clock: ~2 min.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import src  # noqa: F401
from src.eval.golden_set import load_golden_set
from src.llm.openrouter import OpenRouterClient
from src.prompts.loader import load_prompt_by_name
from src.rag.retrievers.classifier_llm import LLMQueryClassifier
from src.rag.retrievers.routing import classify_query, route_for_category

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    p.add_argument("--llm-model", default="openai/gpt-4o-mini")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--output", type=Path, default=Path("data/eval/runs/exp_classifier_dispatch.json")
    )
    args = p.parse_args()

    api_key = os.environ.get("RAG_OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("RAG_OPENROUTER_API_KEY not set")

    classifier_prompt = load_prompt_by_name("classify_query")
    llm = OpenRouterClient(api_key=api_key)
    llm_clf = LLMQueryClassifier(llm=llm, model=args.llm_model, prompt=classifier_prompt)

    golden = load_golden_set(args.golden)
    queries = golden.queries[: args.limit] if args.limit else golden.queries
    print(f"Running on {len(queries)} queries")
    print(f"LLM classifier: {args.llm_model}\n")

    regex_cats: list[str] = []
    llm_cats: list[str] = []
    gold_cats: list[str] = []
    per_query: list[dict[str, Any]] = []
    confusion: defaultdict[tuple[str, str], int] = defaultdict(int)

    for idx, q in enumerate(queries):
        regex_cat = classify_query(q.text)
        llm_cat = await llm_clf.classify(q.text)
        regex_cats.append(regex_cat)
        llm_cats.append(llm_cat)
        gold_cats.append(q.category)
        confusion[(regex_cat, llm_cat)] += 1

        regex_path = route_for_category(regex_cat)
        llm_path = route_for_category(llm_cat)
        marker = " "
        if regex_path != llm_path:
            marker = "*"
        print(
            f"  [{idx + 1:3d}/{len(queries)}] {q.query_id:42s} "
            f"gold={q.category:12s} regex={regex_cat:12s}({regex_path}) "
            f"llm={llm_cat:12s}({llm_path}) {marker}"
        )

        per_query.append(
            {
                "query_id": q.query_id,
                "query": q.text,
                "gold_category": q.category,
                "regex_category": regex_cat,
                "llm_category": llm_cat,
                "regex_path": regex_path,
                "llm_path": llm_path,
                "agree": regex_path == llm_path,
            }
        )

    # ---- Aggregates ----
    n = len(per_query)
    regex_hybrid = sum(1 for r in per_query if r["regex_path"] == "hybrid")
    llm_hybrid = sum(1 for r in per_query if r["llm_path"] == "hybrid")
    agree = sum(1 for r in per_query if r["agree"])

    # Hybrid-rate by GOLD category — the truth
    by_gold_regex_hybrid: defaultdict[str, int] = defaultdict(int)
    by_gold_llm_hybrid: defaultdict[str, int] = defaultdict(int)
    by_gold_n: Counter[str] = Counter()
    for r in per_query:
        gc = r["gold_category"]
        by_gold_n[gc] += 1
        if r["regex_path"] == "hybrid":
            by_gold_regex_hybrid[gc] += 1
        if r["llm_path"] == "hybrid":
            by_gold_llm_hybrid[gc] += 1

    print("\n" + "=" * 72)
    print(f"DISPATCH RATE OVER n={n}")
    print("=" * 72)
    print(f"  regex dispatched to hybrid : {regex_hybrid:4d} / {n}  ({regex_hybrid / n:.1%})")
    print(f"  llm   dispatched to hybrid : {llm_hybrid:4d} / {n}  ({llm_hybrid / n:.1%})")
    print(f"  classifier agreement (path): {agree:4d} / {n}  ({agree / n:.1%})")
    print()
    print("BY GOLD CATEGORY (figure/table/multi_hop SHOULD go hybrid):")
    print(f"  {'gold':14s} {'n':>4s} {'regex→hybrid':>14s} {'llm→hybrid':>14s} {'Δ rel':>10s}")
    for cat in sorted(by_gold_n.keys()):
        n_cat = by_gold_n[cat]
        rx = by_gold_regex_hybrid[cat]
        lm = by_gold_llm_hybrid[cat]
        delta = (lm - rx) / max(n_cat, 1)
        print(
            f"  {cat:14s} {n_cat:4d} {rx:>5d}/{n_cat:<3d}({rx / n_cat:.0%}) "
            f"{lm:>5d}/{n_cat:<3d}({lm / n_cat:.0%}) {delta:+9.1%}"
        )

    print("\nCONFUSION MATRIX (regex_category x llm_category):")
    cats = ["figure", "table", "multi_hop", "factual", "definitional"]
    header = "  " + "regex \\ llm".ljust(14) + " ".join(f"{c:>14s}" for c in cats)
    print(header)
    for rcat in cats:
        row = [f"{confusion[(rcat, ccat)]:>14d}" for ccat in cats]
        print(f"  {rcat:14s} " + " ".join(row))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": {
                    "golden": str(args.golden),
                    "llm_model": args.llm_model,
                    "classifier_prompt_version": classifier_prompt.version,
                },
                "summary": {
                    "n": n,
                    "regex_hybrid_count": regex_hybrid,
                    "llm_hybrid_count": llm_hybrid,
                    "path_agreement": agree,
                },
                "per_query": per_query,
            },
            fh,
            indent=2,
        )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
