"""Bet 2 decisive test, scoring + gate arm: page-recall A/B and selective gate.

Companion to bet2_agentic_mmlb_run.py. That driver produced an agentic run over
the `routing_study` collection; this script scores it against the committed
depth-50 baseline's TEXT leg at PAGE granularity, then runs the selective-gate
policy analysis on page-recall.

Mirrors bet2_selective_gate.py's POLICY logic (baseline / all-agentic /
oracle-category / oracle-perquery) but on a retrieval metric (page recall@k),
not the gemma3:4b answer_correctness judge. The strategic frame (2026-05-29
capstone): end-to-end QA is post-retrieval-bound, so this is a RETRIEVAL-METRIC
result -- "does the decomposition gate lift page-recall on MMLongBench, per
category", NOT an end-to-end-accuracy claim.

Page identity reuses scripts/rescore_mmlb_pages (`_page_of`,
`_dedup_pages_in_rank`, `Page`): the exact `::p(\\d+)` mapping + rank-order dedup
the depth-50 baseline was scored with, so a (paper, page) tuple here is the same
one the baseline self-check graded. recall@k is computed directly (rescore() is
fixed at k=10) so the A/B can report both @5 and @10.

A/B is TEXT-vs-TEXT: baseline `text_top50` vs agentic `agentic_top50`. The
agentic tier only touches text retrieval, so this is the honest comparison; the
visual leg and fusion are out of scope here.

Noise band: per-query recall is fractional (mean ~1.6 relevant pages/query), so
the primary CI is a PAIRED bootstrap on the agentic-minus-baseline delta over the
shared query set (the arms share queries, so a paired resample is tighter and
correct). A delta whose 95% CI straddles 0 is flagged "noise".

Usage:
    .venv/Scripts/python.exe -m scripts.experiments.bet2_mmlb_gate \
        --agentic data/eval/runs/bet2-agentic-<ts>/agentic.json \
        --baseline data/eval/runs/depth50-20260525-015216/depth50.json \
        --golden data/golden/mmlongbench-v1.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from scripts.rescore_mmlb_pages import Page, _dedup_pages_in_rank

# Resample count for the paired bootstrap CI. 10k is plenty for a stable 95%
# interval at n<=107 and runs in well under a second.
_BOOTSTRAP = 10_000
_SEED = 0


def _relevant_by_qid(golden: dict[str, Any]) -> dict[str, set[Page]]:
    """(paper_id, page) relevance set per query, paper-aware -- identical to
    rescore_mmlb_pages.rescore's mapping so the denominator matches the baseline."""
    return {
        q["query_id"]: {
            (q["paper_id"], page)
            for page in (q.get("relevant_pages") or [])
            if q.get("paper_id")
        }
        for q in golden["queries"]
    }


def _recall_at_k(ranked_chunk_ids: list[str], relevant: set[Page], k: int) -> float:
    """Fraction of relevant pages found in the top-k UNIQUE pages of the ranking.

    Dedups chunk-ids to pages FIRST, then truncates to k pages -- a true
    page-recall@k. The committed baseline self-check (driver.log text-only@10
    =0.6184) instead truncated to the top-10 CHUNKS before dedup, so its k=10
    page set can be smaller; that convention reports a slightly lower baseline
    (page-recall@10 here is 0.6324 on the same ranking). Both arms of the A/B go
    through THIS function, so the delta is convention-invariant; the absolute
    baseline is reconciled to the committed number in the printed output.
    """
    if not relevant:
        return 0.0
    pages = _dedup_pages_in_rank(ranked_chunk_ids)[:k]
    hits = sum(1 for p in pages if p in relevant)
    return hits / len(relevant)


def _recall_chunktrunc(ranked_chunk_ids: list[str], relevant: set[Page], k: int) -> float:
    """Driver-convention recall: truncate to top-k CHUNKS first, then dedup to
    pages (what diagnose_depth50_run fed rescore). Only used for the reconcile
    line so the page-recall baseline ties to the committed 0.6184 figure."""
    if not relevant:
        return 0.0
    pages = _dedup_pages_in_rank(ranked_chunk_ids[:k])
    hits = sum(1 for p in pages if p in relevant)
    return hits / len(relevant)


def _load_run(path: Path, ranking_field: str) -> dict[str, dict[str, Any]]:
    """query_id -> {category, ranking} from a run JSON's per_query."""
    d = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for rec in d["per_query"]:
        out[rec["query_id"]] = {
            "category": rec.get("category", ""),
            "ranking": rec.get(ranking_field) or [],
        }
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _paired_bootstrap_ci(
    deltas: list[float], rng: random.Random, *, iters: int = _BOOTSTRAP
) -> tuple[float, float]:
    """95% percentile CI on the MEAN of paired per-query deltas. Resamples
    queries with replacement (paired: the delta already differenced the two arms
    on the same query), so it captures the variance of the per-query improvement."""
    if not deltas:
        return (float("nan"), float("nan"))
    n = len(deltas)
    means: list[float] = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return (lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agentic", type=Path, required=True, help="bet2 agentic run JSON")
    ap.add_argument(
        "--baseline",
        type=Path,
        default=Path("data/eval/runs/depth50-20260525-015216/depth50.json"),
        help="depth-50 baseline run JSON (its text_top50 leg is the A/B baseline)",
    )
    ap.add_argument("--golden", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    ap.add_argument(
        "--agentic-field",
        default="agentic_top50",
        help="per_query field holding the agentic ranking",
    )
    ap.add_argument(
        "--baseline-field",
        default="text_top50",
        help="per_query field holding the baseline TEXT ranking (text-vs-text A/B)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    relevant = _relevant_by_qid(golden)
    agentic = _load_run(args.agentic, args.agentic_field)
    baseline = _load_run(args.baseline, args.baseline_field)
    rng = random.Random(_SEED)

    # In-corpus answerable subset, IDENTICAL to the baseline self-check's n=107:
    # category != out_of_corpus AND a non-empty relevance set. The category
    # guard is load-bearing -- 4 out_of_corpus queries carry a stray
    # relevant_pages annotation (evidence_sources=[], labelled unanswerable), and
    # rescore_mmlb_pages drops them by CATEGORY, not by empty pages. Filtering on
    # relevant_pages alone would re-admit those 4 and inflate n to 111, breaking
    # apples-to-apples with the committed baseline.
    qids = sorted(
        q
        for q in baseline
        if q in agentic and baseline[q]["category"] != "out_of_corpus" and relevant.get(q)
    )
    cats = sorted({baseline[q]["category"] for q in qids})

    # Per-query recall@5 / @10 for each arm.
    rec: dict[str, dict[str, float]] = {}
    for q in qids:
        rel = relevant[q]
        rec[q] = {
            "b5": _recall_at_k(baseline[q]["ranking"], rel, 5),
            "b10": _recall_at_k(baseline[q]["ranking"], rel, 10),
            "a5": _recall_at_k(agentic[q]["ranking"], rel, 5),
            "a10": _recall_at_k(agentic[q]["ranking"], rel, 10),
        }

    by_cat: dict[str, list[str]] = defaultdict(list)
    for q in qids:
        by_cat[baseline[q]["category"]].append(q)

    summary: dict[str, Any] = {
        "n": len(qids),
        "agentic_run": str(args.agentic),
        "baseline_run": str(args.baseline),
        "by_category": {},
        "policies": {},
    }

    # ---- per-category A/B (recall@10 primary, recall@5 shown) + paired CI ----
    print(f"Page-recall A/B: baseline TEXT leg vs agentic, MMLongBench in-corpus n={len(qids)}")
    print(f"  agentic  = {args.agentic}")
    print(f"  baseline = {args.baseline} ({args.baseline_field})\n")
    header = (
        f"  {'category':<10}{'n':>4}"
        f"{'base@5':>9}{'agt@5':>9}{'d@5':>8}"
        f"{'base@10':>9}{'agt@10':>9}{'d@10':>8}"
        f"{'95%CI@10':>18}  verdict"
    )
    print(header)
    gate_cats: set[str] = set()
    for c in cats:
        qs = by_cat[c]
        b5 = _mean([rec[q]["b5"] for q in qs])
        a5 = _mean([rec[q]["a5"] for q in qs])
        b10 = _mean([rec[q]["b10"] for q in qs])
        a10 = _mean([rec[q]["a10"] for q in qs])
        d10_list = [rec[q]["a10"] - rec[q]["b10"] for q in qs]
        ci_lo, ci_hi = _paired_bootstrap_ci(d10_list, rng)
        straddles = ci_lo <= 0.0 <= ci_hi
        sig = "noise" if straddles else "real"
        on = a10 >= b10
        if on:
            gate_cats.add(c)
        verdict = ("gate ON " if on else "gate off") + f" ({sig})"
        ci_str = f"[{ci_lo:+.3f},{ci_hi:+.3f}]"
        print(
            f"  {c:<10}{len(qs):>4}"
            f"{b5:>9.3f}{a5:>9.3f}{a5 - b5:>+8.3f}"
            f"{b10:>9.3f}{a10:>9.3f}{a10 - b10:>+8.3f}"
            f"{ci_str:>18}  {verdict}"
        )
        summary["by_category"][c] = {
            "n": len(qs),
            "base_recall_at_5": b5,
            "agentic_recall_at_5": a5,
            "base_recall_at_10": b10,
            "agentic_recall_at_10": a10,
            "delta_at_10": a10 - b10,
            "delta_at_10_ci95": [ci_lo, ci_hi],
            "delta_at_10_ci_straddles_zero": straddles,
        }

    # overall row
    ob5 = _mean([rec[q]["b5"] for q in qids])
    oa5 = _mean([rec[q]["a5"] for q in qids])
    ob10 = _mean([rec[q]["b10"] for q in qids])
    oa10 = _mean([rec[q]["a10"] for q in qids])
    od10 = [rec[q]["a10"] - rec[q]["b10"] for q in qids]
    oci_lo, oci_hi = _paired_bootstrap_ci(od10, rng)
    o_straddles = oci_lo <= 0.0 <= oci_hi
    oci_str = f"[{oci_lo:+.3f},{oci_hi:+.3f}]"
    print(
        f"  {'ALL':<10}{len(qids):>4}"
        f"{ob5:>9.3f}{oa5:>9.3f}{oa5 - ob5:>+8.3f}"
        f"{ob10:>9.3f}{oa10:>9.3f}{oa10 - ob10:>+8.3f}"
        f"{oci_str:>18}  {'(noise)' if o_straddles else '(real)'}"
    )
    summary["overall"] = {
        "base_recall_at_5": ob5,
        "agentic_recall_at_5": oa5,
        "base_recall_at_10": ob10,
        "agentic_recall_at_10": oa10,
        "delta_at_10": oa10 - ob10,
        "delta_at_10_ci95": [oci_lo, oci_hi],
        "delta_at_10_ci_straddles_zero": o_straddles,
    }

    # Reconcile the page-recall baseline to the committed chunk-truncated number
    # (driver.log text-only@10 = 0.6184). Same ranking, same n; the only
    # difference is dedup-then-truncate (here) vs truncate-then-dedup (driver).
    base_chunktrunc = _mean(
        [_recall_chunktrunc(baseline[q]["ranking"], relevant[q], 10) for q in qids]
    )
    print(
        f"\n  reconcile: baseline page-recall@10={ob10:.4f} (top-10 unique pages); "
        f"committed driver convention (top-10 chunks then dedup)={base_chunktrunc:.4f}"
    )
    summary["baseline_recall_at_10_driver_convention"] = base_chunktrunc

    # ---- policy means on recall@10 ----
    def policy_value(policy: str, q: str) -> float:
        if policy == "baseline":
            use = False
        elif policy == "all-agentic":
            use = True
        elif policy == "oracle-category":
            use = baseline[q]["category"] in gate_cats
        else:  # oracle-perquery (unreachable upper bound)
            use = rec[q]["a10"] > rec[q]["b10"]
        return rec[q]["a10"] if use else rec[q]["b10"]

    print(f"\n  gate categories (agentic recall@10 >= baseline): {sorted(gate_cats)}")
    base_mean = _mean([rec[q]["b10"] for q in qids])
    print(f"\n  {'policy':<18}{'recall@10':>11}{'vs base':>10}")
    for p in ["baseline", "all-agentic", "oracle-category", "oracle-perquery"]:
        m = _mean([policy_value(p, q) for q in qids])
        print(f"  {p:<18}{m:>11.4f}{m - base_mean:>+10.4f}")
        summary["policies"][p] = m
    summary["gate_categories"] = sorted(gate_cats)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
