"""Decompose end-to-end QA failure into retrieval vs post-retrieval (Bet 1).

The naive "oracle 0.55 -> real 0.24, so it's retrieval" reading is wrong: it
conflates whether the gold page was retrieved with what generation does once it
has it. This script separates them at full scale, using the exact pages fed to
the model (the run JSON's `pages`/`papers`) checked against the human gold pages,
joined to the official per-query scores (score_mmlb_qa --scored-out).

For the answerable in-corpus queries it reports, with a single model on both arms
(so model/prompt/path are held fixed and only the pages differ):

  - gold-page-in-top-k rate                 (retrieval success rate)
  - of retrieval successes, the CONVERSION rate (generation got it right)
  - failures split into retrieval-miss vs post-retrieval (page present, wrong)
  - oracle (gold pages) vs real paired means, overall and per category

A high retrieval-success rate with a low conversion rate means the binding
constraint is generation/context-handling, not retrieval recall.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def _load_run_pages(path: Path) -> dict[str, set[tuple[str, int]]]:
    """query_id -> set of (paper, page) the model was actually fed."""
    d = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, set[tuple[str, int]]] = {}
    for r in d["per_query"]:
        out[r["query_id"]] = set(zip(r.get("papers", []), r.get("pages", []), strict=False))
    return out


def _load_scored(path: Path) -> dict[str, dict[str, Any]]:
    return {s["query_id"]: s for s in json.loads(path.read_text(encoding="utf-8"))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--real-run", type=Path, required=True, help="run JSON with fed pages (real retrieval)"
    )
    ap.add_argument(
        "--real-scored",
        type=Path,
        required=True,
        help="score_mmlb_qa --scored-out for the real run",
    )
    ap.add_argument(
        "--oracle-scored",
        type=Path,
        default=None,
        help="optional scored-out for the same-model oracle arm",
    )
    ap.add_argument("--golden", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    gold = {
        q["query_id"]: {(q["paper_id"], p) for p in (q.get("relevant_pages") or [])}
        for q in golden["queries"]
    }
    fed = _load_run_pages(args.real_run)
    real = _load_scored(args.real_scored)
    oracle = _load_scored(args.oracle_scored) if args.oracle_scored else {}

    # Answerable in-corpus = has gold pages and was scored as answerable.
    qids = [q for q in real if gold.get(q) and not real[q].get("is_unanswerable")]

    present = correct_present = miss = correct_miss = 0
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "present": 0, "conv": 0})
    for q in qids:
        has = bool(gold[q] & fed.get(q, set()))
        ok = real[q]["score"] > 0
        cat = real[q].get("category", "")
        by_cat[cat]["n"] += 1
        if has:
            present += 1
            correct_present += ok
            by_cat[cat]["present"] += 1
            by_cat[cat]["conv"] += ok
        else:
            miss += 1
            correct_miss += ok

    n = len(qids)
    real_acc = sum(real[q]["score"] for q in qids) / n if n else float("nan")
    print(f"Bet 1 decomposition  (answerable in-corpus n={n}, real run={args.real_run.name})\n")
    print(f"  real end-to-end ACC (answerable)            : {real_acc:.3f}")
    print(f"  gold page IN real top-k                     : {present}/{n} ({present / n:.0%})")
    print(
        f"    of those, generation CORRECT (conversion) : {correct_present}/{present}"
        f" ({correct_present / present:.0%})"
        if present
        else "    (none present)"
    )
    print(
        f"  gold page MISSED by retrieval               : {miss}/{n}; correct anyway: {correct_miss}"
    )
    miss_fail = miss - correct_miss  # gold-missed queries that ALSO scored 0
    fail = n - (correct_present + correct_miss)
    post = present - correct_present
    print(
        f"\n  of {fail} total failures: {miss_fail} retrieval-miss ({miss_fail / fail:.0%}),"
        f" {post} post-retrieval ({post / fail:.0%})"
        if fail
        else "  no failures"
    )
    print(
        f"  -> {'POST-RETRIEVAL' if post > miss_fail else 'RETRIEVAL'} is the dominant failure mode"
    )

    print("\n  per category (n | gold-present | conversion of present):")
    for c in sorted(by_cat):
        b = by_cat[c]
        conv = f"{b['conv']}/{b['present']}" if b["present"] else "-"
        print(f"    {c:<14} n={b['n']:<3} present={b['present']:<3} conv={conv}")

    summary: dict[str, Any] = {
        "n": n,
        "real_acc": real_acc,
        "gold_present": present,
        "conversion_of_present": correct_present / present if present else None,
        "retrieval_miss": miss,
        "retrieval_miss_fail": miss - correct_miss,
        "post_retrieval_fail": post,
        "dominant": "post-retrieval" if post > (miss - correct_miss) else "retrieval",
    }

    if oracle:
        paired = [q for q in qids if q in oracle]
        o = sum(oracle[q]["score"] for q in paired) / len(paired)
        r = sum(real[q]["score"] for q in paired) / len(paired)
        print(
            f"\n  same-model paired oracle vs real (n={len(paired)}): oracle {o:.3f}  real {r:.3f}  delta {r - o:+.3f}"
        )
        summary["paired"] = {"n": len(paired), "oracle": o, "real": r}

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
