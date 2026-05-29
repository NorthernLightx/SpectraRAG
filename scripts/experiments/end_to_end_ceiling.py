"""Quantify the real-retrieval end-to-end ceiling on MMLongBench-Doc.

The strong-VLM end-to-end QA number is gated by retrieval page-recall: a query
is answerable end-to-end only if its gold page is in the top-k fed to the VLM.
This computes the EXACT real-retrieval page-recall@k of the shipped router (the
fused leg of a depth-50 dump) against the human gold pages — zero GPU, zero
cloud — and combines it with the measured oracle-page generation accuracy to
bound the achievable end-to-end ACC.

It is the grounded, no-cloud form of "finish the end-to-end run": when the cloud
VLM is quota-walled, recall-ceiling x measured-oracle-generation is the tightest
honest statement of where the system sits. The bound is

    E[ACC | answerable] <= recall@k                    (no page -> no answer)
    E[ACC | answerable] ~= recall@k * P(correct | page) (a realistic estimate,
                            with P(correct | page) taken from the oracle-page
                            generation run; distractors in the top-k can only
                            lower it, so this reads as an upper-ish estimate)

Page identity reuses scripts.rescore_mmlb_pages (paper-aware ::pN, dedup in
rank) so a (paper, page) here is the same tuple retrieval is graded on.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from scripts.rescore_mmlb_pages import Page, _dedup_pages_in_rank


def recall_at_k(fused_pages: list[Page], gold: set[Page], k: int) -> float:
    return sum(1 for p in fused_pages[:k] if p in gold) / len(gold)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--retrieval",
        type=Path,
        default=Path("data/eval/runs/depth50-20260525-015216/depth50.json"),
    )
    ap.add_argument("--golden", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    ap.add_argument(
        "--oracle-acc",
        type=float,
        default=0.505,
        help="measured oracle-page generation ACC on the answerable subset",
    )
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50])
    ap.add_argument("--out", type=Path, default=None, help="write a JSON summary here")
    args = ap.parse_args()

    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    gold_by: dict[str, Any] = {}
    for q in golden["queries"]:
        pages = {(q["paper_id"], p) for p in (q.get("relevant_pages") or []) if q.get("paper_id")}
        gold_by[q["query_id"]] = {
            "pages": pages,
            "category": q.get("category", ""),
            "n_pages": len(q.get("relevant_pages") or []),
            "text": q.get("text", ""),
        }

    run = json.loads(args.retrieval.read_text(encoding="utf-8"))
    per_query = run["per_query"] if isinstance(run, dict) else run

    rows: list[tuple[str, str, int, dict[int, float]]] = []
    for rec in per_query:
        qid = rec.get("query_id")
        g = gold_by.get(qid)
        if not g or not g["pages"]:  # need gold pages (excludes OOC / unanswerable)
            continue
        fused = _dedup_pages_in_rank(rec.get("fused_top50") or [])
        rows.append(
            (
                qid,
                g["category"],
                g["n_pages"],
                {k: recall_at_k(fused, g["pages"], k) for k in args.ks},
            )
        )

    cats = sorted({r[1] for r in rows})

    def macro(selector: Callable[..., bool], k: int) -> tuple[float, int]:
        xs = [r[3][k] for r in rows if selector(r)]
        return (sum(xs) / len(xs) if xs else float("nan"), len(xs))

    def fmt_line(name: str, selector: Callable[..., bool]) -> str:
        n = macro(selector, args.ks[0])[1]
        cells = "".join(f"{macro(selector, k)[0]:>9.4f}" for k in args.ks)
        return f"  {name:<14}{n:>5}{cells}"

    print(f"Real-retrieval page-recall@k  (fused router, {args.retrieval.name})")
    print("  answerable in-corpus queries only (those carrying gold relevant_pages)\n")
    header = f"  {'subset':<14}{'n':>5}" + "".join(f"{'@' + str(k):>9}" for k in args.ks)
    print(header)
    print(fmt_line("ALL", lambda r: True))
    for c in cats:
        print(fmt_line(c, lambda r, c=c: r[1] == c))

    print(f"\nOracle-page generation ACC on answerable (measured): {args.oracle_acc:.3f}")
    print("End-to-end ACC estimate on answerable = recall@k x oracle (upper-ish):")
    summary: dict[str, Any] = {
        "recall": {},
        "ceiling": {},
        "oracle_acc": args.oracle_acc,
        "n": len(rows),
    }
    for k in args.ks:
        rk = macro(lambda r: True, k)[0]
        summary["recall"][k] = rk
        if k in (3, 5, 10):
            est = rk * args.oracle_acc
            summary["ceiling"][k] = est
            print(f"  @k={k:<3} recall {rk:.3f} x {args.oracle_acc:.3f} = {est:.3f}")

    # Bet 3: identify the document-wide aggregation / multi-page queries.
    print("\nAggregation / multi-page candidates (n_pages>=3 or text matches 'how many'):")
    agg = [
        (qid, cat, npg, gold_by[qid]["text"])
        for qid, cat, npg, _ in rows
        if npg >= 3 or "how many" in gold_by[qid]["text"].lower()
    ]
    by_paper: dict[str, int] = {}
    for qid, cat, npg, text in sorted(agg, key=lambda x: -x[2]):
        paper = next(iter(gold_by[qid]["pages"]))[0]
        by_paper[paper] = by_paper.get(paper, 0) + 1
        print(f"  {qid:<34} cat={cat:<8} n_pages={npg:<3} {text[:66]}")
    summary["aggregation_n"] = len(agg)
    summary["aggregation_by_paper"] = by_paper
    print(f"\n  {len(agg)} aggregation candidates across {len(by_paper)} papers: {by_paper}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
