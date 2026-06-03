"""Within-doc DENSE control for the grep-recovers-misses probe.

The probe (grep_recovers_misses.py) showed BM25 recovers 19/24 RAG-missed gold
pages, but its attribution used a depth-50-capped proxy for "dense". This closes
that gap: run a true within-document bge-m3 dense ranking over the SAME per-page
text corpus BM25 used, all pages, no depth cap. That isolates the one variable
the proxy couldn't — lexical vs dense — on identical input.

The decisive question: of the queries BM25 recovered but the proxy attributed to
"lexical only", does real within-doc dense ALSO recover them? If yes, grep is
dominated and the lever is SCOPING (use a within-doc dense leg, no tool agent).
If dense misses pages BM25 finds, that residual is the genuine exact-lexical edge.

Local, one GPU model (bge-m3 via Ollama). Reuses the probe's page-text + page-id
helpers so the corpus is byte-identical.

NOT authoring ground truth: gold pages are human MMLongBench labels; this only
re-ranks existing pages with a dense retriever.

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.dense_control_misses \
        --probe data/eval/runs/grep_recovers_misses.json \
        --out data/eval/runs/dense_control_misses.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.grep_recovers_misses import Page, _hit_at_k, _page_texts
from src.embeddings.ollama_bge import OllamaBgeEmbedder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


async def _dense_rank(embedder: OllamaBgeEmbedder, paper_id: str, query: str) -> list[Page] | None:
    """Rank a paper's pages by bge-m3 cosine to the query, over all page text."""
    texts = _page_texts(paper_id)
    if texts is None:
        return None
    # Ollama can 500 on an empty prompt; give blank pages a sentinel so the
    # indices stay aligned with page numbers. A blank page ranks low anyway.
    safe = [t if t.strip() else "(blank page)" for t in texts]
    vecs = await embedder.embed_texts([query, *safe])
    arr = np.asarray(vecs, dtype=np.float64)
    q = arr[0]
    pages = arr[1:]
    qn = q / (np.linalg.norm(q) + 1e-9)
    pn = pages / (np.linalg.norm(pages, axis=1, keepdims=True) + 1e-9)
    sims = pn @ qn
    order = np.argsort(-sims)
    return [(paper_id, int(i) + 1) for i in order]


async def run(args: argparse.Namespace) -> int:
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    rows = probe["rows"]
    embedder = OllamaBgeEmbedder(base_url=args.ollama_url, model=args.model, timeout=args.timeout)

    out_rows: list[dict[str, Any]] = []
    for r in rows:
        paper_id = r["paper_id"]
        gold: set[Page] = {(paper_id, p) for p in r["gold_pages"]}
        ranked = await _dense_rank(embedder, paper_id, r["query"])
        if ranked is None:
            continue
        d5 = _hit_at_k(ranked, gold, 5)
        d10 = _hit_at_k(ranked, gold, 10)
        out_rows.append({**r, "dense_hit_at_5": d5, "dense_hit_at_10": d10})
        print(f"  {r['qid'].split('_')[1]:5} [{r['category']:<7}] bm25@10={int(r['lex_hit_at_10'])} "
              f"dense@10={int(d10)}  {r['query'][:54]}")

    n = len(out_rows)
    bm25_10 = sum(r["lex_hit_at_10"] for r in out_rows) / n
    dense_10 = sum(r["dense_hit_at_10"] for r in out_rows) / n
    dense_5 = sum(r["dense_hit_at_5"] for r in out_rows) / n
    print(f"\nWithin-doc recall on the RAG-missed set (n={n}, same page-text corpus):")
    print(f"  BM25  recall@10 = {bm25_10:.3f}  ({sum(r['lex_hit_at_10'] for r in out_rows)}/{n})")
    print(f"  dense recall@5  = {dense_5:.3f}   recall@10 = {dense_10:.3f}  "
          f"({sum(r['dense_hit_at_10'] for r in out_rows)}/{n})")

    # The clean attribution, depth-cap removed: BM25-only = BM25 hits AND dense misses.
    bm25_wins = [r for r in out_rows if r["lex_hit_at_10"]]
    lexical_only = [r for r in bm25_wins if not r["dense_hit_at_10"]]
    dense_too = [r for r in bm25_wins if r["dense_hit_at_10"]]
    both_miss = [r for r in out_rows if not r["lex_hit_at_10"] and not r["dense_hit_at_10"]]
    dense_only = [r for r in out_rows if r["dense_hit_at_10"] and not r["lex_hit_at_10"]]
    print(f"\n  Clean attribution of the {len(bm25_wins)} BM25 recoveries:")
    print(f"    within-doc dense ALSO recovers : {len(dense_too)}/{len(bm25_wins)}  -> SCOPING (use a dense leg)")
    print(f"    genuine lexical-only edge      : {len(lexical_only)}/{len(bm25_wins)}")
    print(f"  dense recovers where BM25 fails  : {len(dense_only)}")
    print(f"  neither finds it (pixel-only/hard): {len(both_miss)}")
    if lexical_only:
        print("\n  genuine lexical-only (BM25 hit, dense miss) — the real DCI edge:")
        for r in lexical_only:
            ng = len(r["gold_pages"])
            tag = " [multi-gold: recall@10 inflated]" if ng > 1 else ""
            print(f"    {r['qid'].split('_')[1]:5} [{r['category']:<7}] #gold={ng}{tag}  {r['query'][:58]}")

    args.out.write_text(json.dumps({
        "n": n,
        "bm25_recall_at_10": bm25_10,
        "dense_recall_at_5": dense_5, "dense_recall_at_10": dense_10,
        "scoping_recoveries": len(dense_too), "lexical_only": len(lexical_only),
        "dense_only": len(dense_only), "neither": len(both_miss),
        "rows": out_rows,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", type=Path, default=Path("data/eval/runs/grep_recovers_misses.json"))
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=Path("data/eval/runs/dense_control_misses.json"))
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
