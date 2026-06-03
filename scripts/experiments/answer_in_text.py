"""Is the gold ANSWER in the page text, or only in pixels? (agent vs image)

Discriminates two explanations for why text retrieval (grep or dense) misses:

  (1) "the agent is dumb"  -> the answer IS in the page text, retrieval just
      ranked it low; a smarter retriever/agent has room.
  (2) "the data is images" -> the answer is NOT in the extractable text at all
      (chart pixels, captionless figure); no text tool, however smart, can
      retrieve or read it. Only vision can.

For each query it checks whether any `expected_facts` string (the human gold
answer) appears in the gold page's PyMuPDF text. Reported two ways:
  - on the 24 RAG-missed set (the queries under debate), joined to whether
    within-doc BM25 / dense recovered the page;
  - on ALL answerable queries, to size the dataset-level text-vs-pixel ceiling.

A text-presence "hit" is a conservative upper bound on text-retrievability: the
fact tokens being on the page does not guarantee a lexical query would rank it,
but the fact tokens being ABSENT guarantees no text tool can.

NOT authoring ground truth: gold facts are the human MMLongBench labels.

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.answer_in_text \
        --probe data/eval/runs/grep_recovers_misses.json \
        --dense data/eval/runs/dense_control_misses.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.experiments.grep_recovers_misses import _STOP, _page_texts

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_NORM_RE = re.compile(r"[^a-z0-9%]+")


def _norm(s: str) -> str:
    return _NORM_RE.sub(" ", s.lower()).strip()


def _fact_in_text(fact: str, page_text: str) -> bool:
    """Conservative: the normalized fact is a substring, OR >=80% of its content
    words appear on the page. Catches '21%' / 'Berlin School...' alike."""
    nf = _norm(fact)
    nt = _norm(page_text)
    if not nf:
        return False
    if nf in nt:
        return True
    fw = [w for w in nf.split() if len(w) > 1 and w not in _STOP]
    if not fw:
        return False  # fact was all stopwords/symbols; substring check already failed
    hits = sum(1 for w in fw if w in nt)
    return hits / len(fw) >= 0.8


def _answer_in_text(facts: list[str], gold_pages: list[int], paper_id: str) -> bool:
    texts = _page_texts(paper_id) or []
    for p in gold_pages:
        if 1 <= p <= len(texts):
            page_text = texts[p - 1]
            if any(_fact_in_text(f, page_text) for f in facts):
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", type=Path, default=Path("data/eval/runs/grep_recovers_misses.json"))
    ap.add_argument("--dense", type=Path, default=Path("data/eval/runs/dense_control_misses.json"))
    ap.add_argument("--golden", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/runs/answer_in_text.json"))
    args = ap.parse_args()

    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    gmeta = {q["query_id"]: q for q in golden["queries"]}

    # ---- (A) Dataset-level: of all answerable queries, how many are pixel-only? ----
    all_rows = []
    for q in golden["queries"]:
        facts = q.get("expected_facts") or []
        pages = q.get("relevant_pages") or []
        paper = q.get("paper_id")
        if not facts or not pages or not paper:
            continue  # OOC / unlabeled
        in_text = _answer_in_text([str(f) for f in facts], pages, paper)
        all_rows.append({"qid": q["query_id"], "category": q.get("category", ""), "answer_in_text": in_text})
    a_n = len(all_rows)
    a_text = sum(r["answer_in_text"] for r in all_rows)
    print(f"(A) ALL answerable in-corpus (n={a_n}) — is the gold answer in the page TEXT?")
    print(f"    answer in text  : {a_text}/{a_n} ({a_text / a_n:.0%})  <- a text tool/agent could in principle serve")
    print(f"    answer pixels-only: {a_n - a_text}/{a_n} ({(a_n - a_text) / a_n:.0%})  <- only vision can; grep/dense both blind")
    bycat: dict[str, list[bool]] = {}
    for r in all_rows:
        bycat.setdefault(r["category"], []).append(r["answer_in_text"])
    print("    by category (answer-in-text rate):")
    for c in sorted(bycat):
        v = bycat[c]
        print(f"      {c:<10} {sum(v)}/{len(v)} ({sum(v) / len(v):.0%})")

    # ---- (B) The 24 RAG-missed set: agent-room vs image-only ----
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    dense = {r["qid"]: r for r in json.loads(args.dense.read_text(encoding="utf-8"))["rows"]} \
        if args.dense.exists() else {}
    miss_rows: list[dict[str, Any]] = []
    for r in probe["rows"]:
        q = gmeta[r["qid"]]
        facts = [str(f) for f in (q.get("expected_facts") or [])]
        in_text = _answer_in_text(facts, r["gold_pages"], r["paper_id"])
        d = dense.get(r["qid"], {})
        miss_rows.append({
            "qid": r["qid"], "category": r["category"], "query": r["query"],
            "facts": facts, "gold_page_chars": r["gold_page_chars"],
            "answer_in_text": in_text,
            "bm25_hit": r["lex_hit_at_10"], "dense_hit": d.get("dense_hit_at_10"),
        })
    m_n = len(miss_rows)
    m_text = sum(r["answer_in_text"] for r in miss_rows)
    print(f"\n(B) RAG-missed set (n={m_n}) — why text retrieval missed:")
    print(f"    answer IN page text (agent/retriever has room): {m_text}/{m_n}")
    print(f"    answer ONLY in pixels (no text tool can ever)  : {m_n - m_text}/{m_n}")
    print("\n    pixels-only missed queries (claim 2 — needs vision, not a smarter grep):")
    for r in miss_rows:
        if not r["answer_in_text"]:
            print(f"      {r['qid'].split('_')[1]:5} [{r['category']:<7}] chars={r['gold_page_chars']:<5}"
                  f" gold={r['facts']!s:.40}  {r['query'][:46]}")

    args.out.write_text(json.dumps({
        "all_answerable": {"n": a_n, "answer_in_text": a_text, "rows": all_rows},
        "rag_missed": {"n": m_n, "answer_in_text": m_text, "rows": miss_rows},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
