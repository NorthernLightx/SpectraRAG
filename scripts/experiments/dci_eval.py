"""Evaluate DCI agentic retrieval on a BRIGHT domain vs a BM25 floor.

Two methods, same corpus + metric (nDCG@10, excluded_ids removed):
  --method bm25  : rank_bm25 over the corpus. Calibration — should land near the
                   paper's published BM25 (biology 18.9); if it does, the corpus
                   and scorer are correct and the DCI number is trustworthy.
  --method dci   : the DciAgent (SEARCH/GREP/READ -> RANK) driving an Ollama model.

Published bars (biology, nDCG@10): BM25 18.9 | dense ~30 | ReasonRank-32B 58.2 |
DCI-Lite (GPT-5.4-nano) 60.0 | DCI-CC (Sonnet 4.6) 77.1.

DCI runs cache per-query to --out and resume, so a cloud run interrupted by the
Ollama quota wall continues without re-spending calls.

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.dci_eval --method bm25
    .venv/Scripts/python.exe -m scripts.experiments.dci_eval --method dci \
        --model gemma3:4b --limit 15 --out data/eval/runs/dci_bio_gemma.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

import src  # noqa: F401 -- loads .env (RAG_OPENROUTER_API_KEY)
from src.dci.agent import DciAgent
from src.dci.tools import CorpusTools
from src.llm.ollama_chat import OllamaChatClient
from src.llm.openrouter import OpenRouterClient
from src.llm.protocol import LLMClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_TOK = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> list[str]:
    # Naive tokenisation: this BM25 is a weak floor (~8-9 nDCG@10) vs the paper's
    # Lucene-class BM25 (18.9). Stopword/stem preprocessing barely moved it — the
    # gap is the BM25 implementation, not tokenisation — so the published 18.9 is
    # the floor to cite; the DCI agent (which never uses this) is scored on the
    # same corpus/gold/metric, so its number is comparable to the published bars.
    return [t for t in _TOK.findall(text.lower()) if len(t) > 1]

_BARS = "bars: BM25 18.9 | dense ~30 | ReasonRank-32B 58.2 | DCI-Lite 60.0 | DCI-CC 77.1"


def _load_corpus(d: Path) -> dict[str, str]:
    docs: dict[str, str] = {}
    with (d / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            docs[o["id"]] = o["content"]
    return docs


def _load_queries(d: Path) -> list[dict[str, Any]]:
    with (d / "queries.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _ndcg_at_k(ranked: list[str], gold: set[str], excluded: set[str], k: int = 10) -> float:
    ranked = [d for d in ranked if d not in excluded][:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def _summary(scores: list[float], label: str) -> None:
    avg = 100 * sum(scores) / len(scores) if scores else 0.0
    print(f"\n{label}: nDCG@10 = {avg:.1f}  (n={len(scores)})\n{_BARS}")


def _run_bm25(corpus: dict[str, str], queries: list[dict[str, Any]], limit: int) -> None:
    ids = list(corpus)
    bm25 = BM25Okapi([_tokens(corpus[i]) for i in ids])
    qs = queries[:limit] if limit else queries
    scores: list[float] = []
    for q in qs:
        sc = bm25.get_scores(_tokens(q["query"]))  # one full scan per query
        order = sorted(range(len(ids)), key=lambda j: sc[j], reverse=True)[:50]
        ranked = [ids[j] for j in order]
        scores.append(_ndcg_at_k(ranked, set(q["gold_ids"]), set(q["excluded_ids"])))
    _summary(scores, "BM25")


def _run_search_only(corpus: dict[str, str], queries: list[dict[str, Any]], limit: int) -> None:
    """No-brain lexical floor: rank by ONE CorpusTools.search on the raw query, no
    agent. The gap between this and the DCI agent is the agentic contribution
    (term selection + multi-step), isolating it from the tool's raw quality."""
    tools = CorpusTools(corpus)
    qs = queries[:limit] if limit else queries
    scores: list[float] = []
    for q in qs:
        ranked = [h.doc_id for h in tools.search(q["query"], top_k=10)]
        scores.append(_ndcg_at_k(ranked, set(q["gold_ids"]), set(q["excluded_ids"])))
    _summary(scores, "SEARCH-only (raw query, no agent)")


def _build_llm(args: argparse.Namespace) -> LLMClient:
    if args.provider == "openrouter":
        key = os.environ.get("RAG_OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("RAG_OPENROUTER_API_KEY not set")
        return OpenRouterClient(api_key=key)
    return OllamaChatClient(base_url=args.ollama)


async def _run_dci(corpus: dict[str, str], queries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    tools = CorpusTools(corpus)
    llm = _build_llm(args)
    agent = DciAgent(tools, llm, args.model, max_steps=args.max_steps,
                     search_k=args.search_k, toolset=args.toolset)

    cache: dict[str, Any] = {}
    if args.out and args.out.exists():
        cache = json.loads(args.out.read_text(encoding="utf-8")).get("per_query", {})

    qs = queries[:args.limit] if args.limit else queries
    scores: list[float] = []
    for i, q in enumerate(qs, 1):
        qid = q["qid"]
        if qid in cache:
            rec = cache[qid]
        else:
            try:
                res = await agent.run(q["query"], mode="retrieval", top_k=10)
            except Exception as exc:  # cloud quota 429 / transport — save progress, skip
                print(f"[{i}/{len(qs)}] {qid}: ERROR {type(exc).__name__}: {str(exc)[:80]} — skipping")
                break
            rec = {"ranked": res.ranked_doc_ids, "n_steps": len(res.steps),
                   "stopped": res.stopped, "tokens_in": res.tokens_in, "tokens_out": res.tokens_out}
            cache[qid] = rec
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps({"model": args.model, "per_query": cache}, indent=2), encoding="utf-8")
        nd = _ndcg_at_k(rec["ranked"], set(q["gold_ids"]), set(q["excluded_ids"]))
        scores.append(nd)
        print(f"[{i}/{len(qs)}] {qid}: nDCG@10={nd:.3f} steps={rec['n_steps']} stopped={rec['stopped']}")
    tin = sum(r.get("tokens_in", 0) for r in cache.values())
    tout = sum(r.get("tokens_out", 0) for r in cache.values())
    print(f"tokens: in={tin:,} out={tout:,}")
    _summary(scores, f"DCI ({args.model})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/dci/bright_biology"))
    ap.add_argument("--method", choices=["bm25", "dci", "search"], default="bm25")
    ap.add_argument("--provider", choices=["ollama", "openrouter"], default="ollama")
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--limit", type=int, default=0, help="first N queries (0=all)")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--search-k", type=int, default=8)
    ap.add_argument("--toolset", choices=["readgrep", "fullbash"], default="fullbash")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    corpus = _load_corpus(args.corpus_dir)
    queries = _load_queries(args.corpus_dir)
    print(f"corpus={len(corpus)} docs, queries={len(queries)}, method={args.method}")
    if args.method == "bm25":
        _run_bm25(corpus, queries, args.limit)
    elif args.method == "search":
        _run_search_only(corpus, queries, args.limit)
    else:
        asyncio.run(_run_dci(corpus, queries, args))


if __name__ == "__main__":
    main()
