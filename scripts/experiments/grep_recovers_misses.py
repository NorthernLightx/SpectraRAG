"""Does lexical grep recover the pages multimodal RAG missed? (DCI steelman)

The claim under test: "agentic grep/bash retrieval is more precise than RAG."
The honest place that could be true is the subset where the shipped fused
retriever (text dense + visual ColQwen2, RRF) FAILS to surface the gold page.
If a plain exact-lexical retriever (BM25 over the document's own page text)
ranks that missed gold page into the top-k, then grep-style retrieval has real
headroom there. If it cannot — because the gold page is a captionless chart
whose answer lives in pixels, not the text layer — then grep is dead on exactly
the queries RAG already can't serve, and only the visual leg can.

This is a fully local, no-GPU probe: committed depth-50 retrieval dump +
PyMuPDF page text + BM25. No Qdrant, no Ollama, no model calls.

It reproduces the committed fused recall@10 first (calibration sanity: if the
page-identity logic is right, fused recall@10 ~= 0.745 from the 2026-05-29
agenda), then partitions the RAG-missed queries into grep-recoverable vs
pixel-only.

NOT authoring ground truth: gold pages are the human MMLongBench labels; this
only re-ranks existing pages with a different retriever and reports recall.

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.grep_recovers_misses \
        --dump data/eval/runs/depth50-20260525-015216/depth50.json \
        --out data/eval/runs/grep_recovers_misses.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import yaml
from rank_bm25 import BM25Okapi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_PDF_ROOT = Path("data/mmlongbench/documents")
_PAGE_RE = re.compile(r"::p(\d+)")
_TOK_RE = re.compile(r"[a-z0-9]+")

# A small stoplist + the MMLongBench question-boilerplate that carries no
# document signal ("according to the chart on page N, ...").
_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "by", "with", "at", "as", "it", "its", "this", "that",
    "these", "those", "from", "what", "which", "who", "whom", "how", "many",
    "much", "does", "do", "did", "page", "according", "shown", "show", "figure",
    "table", "chart", "based", "given", "following", "value", "values", "number",
}

Page = tuple[str, int]


def _page_of(chunk_id: str) -> Page | None:
    m = _PAGE_RE.search(chunk_id)
    if m is None:
        return None
    return chunk_id.split("::", 1)[0], int(m.group(1))


def _dedup_pages(chunk_ids: list[str]) -> list[Page]:
    seen: set[Page] = set()
    out: list[Page] = []
    for cid in chunk_ids:
        p = _page_of(cid)
        if p is None or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _tokens(text: str) -> list[str]:
    return [t for t in _TOK_RE.findall(text.lower()) if len(t) > 1 and t not in _STOP]


def _hit_at_k(pages: list[Page], gold: set[Page], k: int) -> bool:
    return bool(gold & set(pages[:k]))


def _scoped_leg_hit(pq: dict[str, Any], key: str, paper_id: str, gold: set[Page], k: int = 10) -> bool:
    """Within-document recall control: re-rank a committed leg restricted to the
    gold paper, check the top-k. Depth-50-capped (a gold chunk beyond corpus
    rank 50 is invisible), so it can only UNDER-count dense recoveries."""
    scoped = [pg for pg in _dedup_pages(pq[key]) if pg[0] == paper_id]
    return _hit_at_k(scoped, gold, k)


_pdf_cache: dict[str, list[str]] = {}


def _page_texts(paper_id: str) -> list[str] | None:
    """1-indexed page text for a paper; cached. None if the PDF is missing."""
    if paper_id in _pdf_cache:
        return _pdf_cache[paper_id]
    pdf = _PDF_ROOT / f"{paper_id}.pdf"
    if not pdf.exists():
        return None
    with fitz.open(pdf) as doc:
        texts = [page.get_text() for page in doc]
    _pdf_cache[paper_id] = texts
    return texts


def _bm25_rank(paper_id: str, query: str) -> list[Page] | None:
    """Rank a paper's pages by BM25 over their text. Returns (paper, page#)
    in descending score order (1-based page numbers), or None if no PDF."""
    texts = _page_texts(paper_id)
    if texts is None:
        return None
    corpus = [_tokens(t) for t in texts]
    if not any(corpus):
        return []
    bm25 = BM25Okapi([c or ["__empty__"] for c in corpus])
    scores = bm25.get_scores(_tokens(query))
    order = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)
    return [(paper_id, i + 1) for i in order]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=Path("data/eval/runs/depth50-20260525-015216/depth50.json"))
    ap.add_argument("--golden", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    ap.add_argument("--leg", choices=["fused", "text", "visual"], default="fused",
                    help="which RAG ranking counts as 'what RAG retrieved'")
    ap.add_argument("--miss-k", type=int, default=10, help="gold-not-in-top-K defines a RAG miss")
    ap.add_argument("--out", type=Path, default=Path("data/eval/runs/grep_recovers_misses.json"))
    args = ap.parse_args()

    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    gmeta = {q["query_id"]: q for q in golden["queries"]}
    gold_pages = {
        q["query_id"]: {(q["paper_id"], p) for p in (q.get("relevant_pages") or [])}
        for q in golden["queries"]
        if q.get("paper_id") and (q.get("relevant_pages") or [])
    }

    dump = json.loads(args.dump.read_text(encoding="utf-8"))
    leg_key = f"{args.leg}_top50"

    # ---- Calibration: reproduce committed fused recall@k over all answerable ----
    answerable = [pq for pq in dump["per_query"] if gold_pages.get(pq["query_id"])]
    def _recall_at(pq: dict[str, Any], k: int) -> float:
        gold = gold_pages[pq["query_id"]]
        hits = sum(1 for p in _dedup_pages(pq[leg_key])[:k] if p in gold)
        return hits / len(gold)
    base_r5 = sum(_recall_at(pq, 5) for pq in answerable) / len(answerable)
    base_r10 = sum(_recall_at(pq, 10) for pq in answerable) / len(answerable)
    print(f"Calibration ({args.leg} leg, n={len(answerable)} answerable in-corpus):")
    print(f"  RAG recall@5  = {base_r5:.3f}   RAG recall@10 = {base_r10:.3f}")
    print("  (2026-05-29 agenda committed fused recall@5 0.659 / @10 0.745)\n")

    # ---- The RAG-missed set: gold not in top-(miss_k) of the chosen leg ----
    rows: list[dict[str, Any]] = []
    for pq in answerable:
        qid = pq["query_id"]
        gold = gold_pages[qid]
        rag_pages = _dedup_pages(pq[leg_key])
        if _hit_at_k(rag_pages, gold, args.miss_k):
            continue  # RAG already has it; not in the missed set
        q = gmeta[qid]
        paper_id = q["paper_id"]
        query_text = q["text"]
        lex = _bm25_rank(paper_id, query_text)
        if lex is None:
            continue  # no PDF (shouldn't happen; all 20 present)
        # Attribution control: was the miss about LEXICAL-vs-dense, or about
        # corpus-wide-vs-scoped? Re-rank each committed leg restricted to the
        # gold paper (see _scoped_leg_hit) and check the within-doc top-10.
        texts = _page_texts(paper_id) or []
        # gold-page text-presence signal: chars on the gold page(s) and how
        # many distinct query content-terms appear there.
        qterms = set(_tokens(query_text))
        gold_page_nums = [p for (pp, p) in gold if pp == paper_id]
        gp_chars = max((len(texts[p - 1]) for p in gold_page_nums if 1 <= p <= len(texts)), default=0)
        gp_qterms = max(
            (len(qterms & set(_tokens(texts[p - 1]))) for p in gold_page_nums if 1 <= p <= len(texts)),
            default=0,
        )
        rows.append({
            "qid": qid,
            "category": q.get("category", ""),
            "paper_id": paper_id,
            "query": query_text,
            "gold_pages": sorted(gold_page_nums),
            "lex_hit_at_5": _hit_at_k(lex, gold, 5),
            "lex_hit_at_10": _hit_at_k(lex, gold, 10),
            "scoped_text_hit_at_10": _scoped_leg_hit(pq, "text_top50", paper_id, gold),
            "scoped_visual_hit_at_10": _scoped_leg_hit(pq, "visual_top50", paper_id, gold),
            "scoped_fused_hit_at_10": _scoped_leg_hit(pq, "fused_top50", paper_id, gold),
            "gold_page_chars": gp_chars,
            "gold_page_qterms": gp_qterms,
            "n_qterms": len(qterms),
        })

    n = len(rows)
    if n == 0:
        print("No RAG-missed queries — nothing to recover.")
        return
    rec5 = sum(r["lex_hit_at_5"] for r in rows) / n
    rec10 = sum(r["lex_hit_at_10"] for r in rows) / n
    print(f"RAG-missed set (gold not in {args.leg} top-{args.miss_k}): n={n}\n")
    print("  Can lexical BM25 (the grep steelman) recover the missed gold page?")
    print(f"    BM25 recall@5  on missed set = {rec5:.3f}  ({sum(r['lex_hit_at_5'] for r in rows)}/{n})")
    print(f"    BM25 recall@10 on missed set = {rec10:.3f}  ({sum(r['lex_hit_at_10'] for r in rows)}/{n})")

    # Actionable: union recall if you bolted a BM25 leg onto the fused ranking.
    # (Approx: a query is recovered if BM25 top-10 hits, counted as full recall.)
    union10 = sum(
        1 for pq in answerable
        if _hit_at_k(_dedup_pages(pq[leg_key]), gold_pages[pq["query_id"]], 10)
        or any(r["qid"] == pq["query_id"] and r["lex_hit_at_10"] for r in rows)
    ) / len(answerable)
    # binary gold-present@10 for the fused leg, same denominator
    fused_present10 = sum(
        1 for pq in answerable
        if _hit_at_k(_dedup_pages(pq[leg_key]), gold_pages[pq["query_id"]], 10)
    ) / len(answerable)
    print(f"\n  gold-present@10  fused-only = {fused_present10:.3f}  ->  fused+BM25 = {union10:.3f}"
          f"  (+{union10 - fused_present10:.3f})")

    # Partition by category + the pixel-only signal.
    print("\n  by category (missed n | BM25 recovers@10 | median gold-page chars):")
    bycat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        bycat[r["category"]].append(r)
    for c in sorted(bycat):
        rs = bycat[c]
        rec = sum(x["lex_hit_at_10"] for x in rs)
        chars = sorted(x["gold_page_chars"] for x in rs)
        med = chars[len(chars) // 2]
        print(f"    {c:<10} n={len(rs):<3} recover={rec}/{len(rs):<3} median_gold_chars={med}")

    # Attribution: split BM25's recoveries into "scoping would've done it too"
    # (any scoped dense/visual leg also hits) vs "only lexical found it".
    def scoped_any(r: dict[str, Any]) -> bool:
        return bool(r["scoped_text_hit_at_10"] or r["scoped_visual_hit_at_10"] or r["scoped_fused_hit_at_10"])
    bm25_wins = [r for r in rows if r["lex_hit_at_10"]]
    lexical_only = [r for r in bm25_wins if not scoped_any(r)]
    scoping_too = [r for r in bm25_wins if scoped_any(r)]
    print("\n  Attribution of BM25's recoveries (lexical vs mere scoping):")
    print(f"    scoped dense/visual ALSO recovers : {len(scoping_too)}/{len(bm25_wins)}  -> the lever is SCOPING")
    print(f"    only BM25 (lexical) recovers      : {len(lexical_only)}/{len(bm25_wins)}  -> the lever is LEXICAL (DCI)")
    print("    (scoped legs are depth-50-capped; a clean dense control needs within-doc bge-m3 over all pages)")
    print("    lexical-only examples:")
    for r in lexical_only[:6]:
        print(f"      {r['qid'].split('_')[1]:5} [{r['category']:<7}] {r['query'][:64]}")

    pixel_only = [r for r in rows if not r["lex_hit_at_10"]]
    print(f"\n  PIXEL-ONLY (BM25 can't find gold page even @10): {len(pixel_only)}/{n}")
    print("    examples:")
    for r in pixel_only[:6]:
        print(f"      {r['qid'].split('_')[1]:5} [{r['category']:<7}] gold_chars={r['gold_page_chars']:<5}"
              f" qterms_on_gold={r['gold_page_qterms']}/{r['n_qterms']}  {r['query'][:60]}")

    args.out.write_text(json.dumps({
        "leg": args.leg, "miss_k": args.miss_k,
        "calibration": {"recall_at_5": base_r5, "recall_at_10": base_r10, "n_answerable": len(answerable)},
        "missed_n": n,
        "bm25_recall_at_5": rec5, "bm25_recall_at_10": rec10,
        "fused_present_at_10": fused_present10, "fused_plus_bm25_at_10": union10,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
