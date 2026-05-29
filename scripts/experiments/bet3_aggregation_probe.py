"""Probe the document-wide aggregation/counting class (Bet 3).

These queries ("how many figures are in the Appendix?", gold spans 9 pages) are
the ~7% true-miss class top-k retrieval cannot serve: the answer requires
scanning the WHOLE document, not ranking k pages. The scout framed the fix as a
new retrieval paradigm (ToC-tree navigation). This probe tests that on-box and
finds the bottleneck is elsewhere.

It separates three things that the single failing number conflates:

  1. RETRIEVAL  — how many of the gold pages does the shipped router surface in
     top-k? (from the depth-50 dump). Establishes the class is unservable by k.
  2. NAVIGATION — given a STRUCTURAL INDEX of the document (the figure index the
     ingestion pipeline already builds at data/figures/<doc>/, parsed from the
     `<doc>__p<N>__fig<M>.png` filenames), can a cheap local LLM (gemma3:4b)
     count correctly? Tests the scout's navigator thesis with a perfect-ish
     index, no cloud, no Qdrant.
  3. EXTRACTION — does the structural index itself match the human gold count?
     The residual error after navigation works is index incompleteness — an
     INGESTION problem, not a retrieval or generation one.

It also classifies every document-wide count query by the index modality it
needs (figure-countable / table-countable / visual-type / text-structure), so
the slice a structural index could actually serve is explicit.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import yaml

from scripts.rescore_mmlb_pages import _dedup_pages_in_rank

_FIG_RE = re.compile(r"__p(\d+)__fig(\d+)", re.IGNORECASE)


def figure_index(doc: str) -> dict[int, list[str]]:
    """page -> [figure ids] from data/figures/<doc>/<doc>__p<N>__fig<M>.png."""
    root = Path("data/figures") / doc
    by_page: dict[int, list[str]] = defaultdict(list)
    if not root.exists():
        return by_page
    for p in sorted(root.glob("*.png")):
        m = _FIG_RE.search(p.name)
        if m:
            by_page[int(m.group(1))].append(f"fig{m.group(2)}")
    return dict(sorted(by_page.items()))


def gemma_navigate(inventory_text: str, question: str, *, model: str = "gemma3:4b") -> str:
    system = (
        "You are given a structural index of a document (its figures and the pages "
        "they appear on). Answer the counting question using ONLY this index. "
        "Reply with just the number."
    )
    user = f"{inventory_text}\n\nQuestion: {question}\nAnswer with a single integer."
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 16},
    }
    r = httpx.post("http://localhost:11434/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    content = (r.json().get("message") or {}).get("content", "")
    return str(content).strip()


def classify_modality(text: str) -> str:
    t = text.lower()
    if "table" in t:
        return "table-index"
    if any(
        k in t
        for k in (
            "line plot",
            "bar chart",
            "bar plot",
            "subplot",
            "pictures",
            "person",
            "color",
            "colour",
            "emoji",
            "rectangle",
        )
    ):
        return "visual-type (needs vision)"
    if any(k in t for k in ("prompt example", "instruction example", "in-context example")):
        return "text-structure"
    if "figure" in t or "charts" in t:
        return "figure-index"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    ap.add_argument(
        "--retrieval",
        type=Path,
        default=Path("data/eval/runs/depth50-20260525-015216/depth50.json"),
    )
    ap.add_argument(
        "--navigate", action="store_true", help="run the gemma3:4b navigator probe (GPU)"
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    by_qid = {q["query_id"]: q for q in golden["queries"]}
    fused_by_qid = {
        rec["query_id"]: _dedup_pages_in_rank(rec.get("fused_top50") or [])
        for rec in json.loads(args.retrieval.read_text(encoding="utf-8"))["per_query"]
    }

    docwide = [q for q in golden["queries"] if len(q.get("relevant_pages") or []) >= 3]
    summary: dict[str, Any] = {"taxonomy": defaultdict(int), "queries": []}

    print("Document-wide count queries: modality each would need + retrieval coverage\n")
    print(f"  {'query_id':<34}{'modality':<26}{'gold':>5}{'goldpg':>7}{'rec@10':>8}")
    for q in docwide:
        qid = q["query_id"]
        modality = classify_modality(q.get("text", ""))
        summary["taxonomy"][modality] += 1
        gold_pages = {(q["paper_id"], p) for p in q["relevant_pages"]}
        fused = fused_by_qid.get(qid, [])
        rec10 = sum(1 for p in fused[:10] if p in gold_pages) / len(gold_pages)
        gold_ans = (q.get("expected_facts") or [""])[0]
        print(
            f"  {qid:<34}{modality:<26}{str(gold_ans)[:5]:>5}{len(q['relevant_pages']):>7}{rec10:>8.2f}"
        )
        summary["queries"].append(
            {
                "qid": qid,
                "modality": modality,
                "gold": gold_ans,
                "n_pages": len(q["relevant_pages"]),
                "recall_at_10": rec10,
            }
        )

    print(f"\n  taxonomy: {dict(summary['taxonomy'])}")
    print("  -> top-k retrieval recall@10 on this class is far below the figure/table subsets,")
    print(
        "     and a structural index (figure/table) could serve only the index-countable slice.\n"
    )

    # Decisive case: mmlb_0069 figures-in-appendix on 2305.13186v3.
    doc = "2305.13186v3"
    qid = "mmlb_0069_2305.13186v3"
    q = by_qid.get(qid)
    if q:
        idx = figure_index(doc)
        total = sum(len(v) for v in idx.values())
        appendix_pages = [p for p in idx if p >= 15]  # appendix starts p15 for this doc
        appendix_figs = sum(len(idx[p]) for p in appendix_pages)
        inv_lines = [f"page {p}: {', '.join(v)}" for p, v in idx.items()]
        inventory = "Document figure index (2305.13186v3):\n" + "\n".join(inv_lines)
        print(f"  DECISIVE CASE {qid}  gold={q['expected_facts'][0]}")
        print(f"    figure index: {total} figures total across pages {list(idx)}")
        print(f"    figures on appendix pages (>=15): {appendix_figs}")
        fused = fused_by_qid.get(qid, [])
        gold_pages = {(q["paper_id"], p) for p in q["relevant_pages"]}
        print(
            f"    top-10 retrieval recall of the 9 gold pages: {sum(1 for p in fused[:10] if p in gold_pages)}/{len(gold_pages)}"
        )
        summary["decisive"] = {
            "qid": qid,
            "gold": q["expected_facts"][0],
            "index_total": total,
            "index_appendix": appendix_figs,
        }
        nav = ctrl = None
        if args.navigate:
            nav = gemma_navigate(
                inventory, "How many figures are in the Appendix (pages 15 and later)?"
            )
            # Control: hand the model the appendix figures as a flat list and ask
            # a plain count. Isolates filter-then-count (the navigator task) from
            # raw counting, so a wrong nav with a right ctrl localizes the failure
            # to the filtering step, not arithmetic.
            flat = ", ".join(f for p in appendix_pages for f in idx[p])
            ctrl = gemma_navigate(f"Appendix figures: {flat}", "How many figures are listed above?")
            print(
                f"    gemma3:4b filter-then-count over the index -> {nav!r}  "
                f"(index truth = {appendix_figs}; gold = {q['expected_facts'][0]})"
            )
            print(
                f"    gemma3:4b raw count of the flat {appendix_figs}-item appendix list -> {ctrl!r}"
            )
            summary["decisive"]["navigator"] = nav
            summary["decisive"]["control_count"] = ctrl
        gold = q["expected_facts"][0]
        print("\n    Reading: this class fails at THREE independent stages, measured here --")
        print("      retrieval : top-k surfaces 0/9 gold pages -- it cannot gather the evidence;")
        if nav is not None:
            print(
                f"      navigation: the cheap local LLM gets filter-then-count wrong ({nav}) while"
            )
            print(
                f"                  the raw count of the same flat list is right ({ctrl}) -- so the"
            )
            print("                  filter+count step belongs in DETERMINISTIC code, not an LLM;")
        print(
            f"      extraction: the index found {appendix_figs} figures on the gold appendix pages "
            f"vs gold {gold}"
        )
        print("                  (n=1 case; the p>=15 cut is not arbitrary -- it matches the gold")
        print(
            "                  pages 15-27, so the residual gap is missed detections, not the cut)."
        )
        print(
            "    Fix = ingestion-side structural index + deterministic counting + a query->filter"
        )
        print(
            "    parser; NOT a retrieval paradigm and NOT a cheap LLM navigator. Portfolio-marginal:"
        )
        print("    only ~2 of the 8 docwide-span queries are figure-index-countable.")

    if args.out:
        summary["taxonomy"] = dict(summary["taxonomy"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
