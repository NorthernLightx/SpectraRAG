"""Selective agentic decomposition, oracle-gated (ADR 0019 follow-up).

ADR 0019 measured the agentic (decompose -> retrieve-per-subquery -> RRF) tier
on golden v3: figure improved, factual/table regressed, overall within noise.
The open lever the ADR named: fire decomposition ONLY on the queries it helps,
keeping the figure gain without the factual/table cost.

This re-scores that lever WITHOUT new retrieval, from the committed per-query
retrieval metrics of the agentic run and its matched non-agentic baseline (same
v3 set, same retriever=pipeline, rerank=False):

  agentic  = data/eval/agentic-text-only.json   (run fd50bbda0212)
  baseline = data/eval/baseline-text-only.json  (run 325375af3043)

It reports, per category and overall, the macro retrieval metrics under three
policies:
  - baseline            (decompose never)
  - all-agentic         (decompose always; the ADR 0019 run)
  - oracle category-gate (decompose only on categories where agentic >= baseline
                          on the primary metric -- the realistic gate a cheap
                          per-query classifier would approximate)
  - oracle per-query     (decompose only where it improves THAT query -- the
                          unreachable upper bound, to size the gap a classifier
                          would have to close)

Caveat carried forward (the strategist's): this is golden v3 (text-heavy arXiv,
n=39). Whether the gate transfers to MMLongBench needs a fresh agentic run on
that corpus, which is Qdrant-gated (collection offline this session). This
validates the MECHANISM on the data we can touch, not the MMLongBench magnitude.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Any

# ADR 0019's verdict rests on answer_correctness, NOT retrieval nDCG: chunk-ids
# changed in ADR 0017 and v3's relevant_chunk_ids were never re-anchored, so the
# stored retrieval.* metrics are ~0 by construction. answer_correctness (the
# gemma3:4b judge vs expected_facts) is the chunk-id-robust scoreboard.
_METRIC = "generation.answer_correctness"
# ADR 0016 calibrated the gemma3:4b judge at +/-0.07 std at n=40; noise on a
# subset of size n scales ~ 0.07 * sqrt(40/n).
_JUDGE_STD_AT_40 = 0.07


def _dotted(rec: dict[str, Any], path: str) -> Any:
    cur: Any = rec
    for part in path.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def _load(path: Path, metric: str) -> dict[str, dict[str, Any]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for rec in d["per_query"]:
        out[rec["query_id"]] = {
            "category": rec.get("category", ""),
            "score": _dotted(rec, metric),
        }
    return out


def _macro(scores: list[Any]) -> tuple[float, int]:
    xs = [s for s in scores if isinstance(s, (int, float))]
    return (sum(xs) / len(xs) if xs else float("nan"), len(xs))


def _noise(n: int) -> float:
    return _JUDGE_STD_AT_40 * sqrt(40 / n) if n else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agentic", type=Path, default=Path("data/eval/agentic-text-only.json"))
    ap.add_argument("--baseline", type=Path, default=Path("data/eval/baseline-text-only.json"))
    ap.add_argument("--metric", default=_METRIC, help="dotted per-query metric path")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    agentic = _load(args.agentic, args.metric)
    baseline = _load(args.baseline, args.metric)
    # in-corpus only: a query is scorable where BOTH arms have a numeric metric.
    qids = [
        q
        for q in baseline
        if q in agentic
        and isinstance(baseline[q]["score"], (int, float))
        and isinstance(agentic[q]["score"], (int, float))
    ]
    cats = sorted({baseline[q]["category"] for q in qids})

    by_cat: dict[str, list[str]] = defaultdict(list)
    for q in qids:
        by_cat[baseline[q]["category"]].append(q)

    # Derive the gate set from the data: fire agentic on categories where it
    # does not regress. A delta inside the judge noise band is flagged so a
    # reader sees it is not a real signal.
    gate_cats: set[str] = set()
    print(f"Per-category agentic vs baseline ({args.metric}), v3 in-corpus n={len(qids)}\n")
    print(f"  {'category':<14}{'n':>4}{'base':>9}{'agentic':>9}{'delta':>9}{'noise+-':>9}  verdict")
    summary: dict[str, Any] = {
        "metric": args.metric,
        "by_category": {},
        "policies": {},
        "n": len(qids),
    }
    for c in cats:
        qs = by_cat[c]
        b, _ = _macro([baseline[q]["score"] for q in qs])
        a, _ = _macro([agentic[q]["score"] for q in qs])
        nz = _noise(len(qs))
        sig = "real" if abs(a - b) > nz else "noise"
        on = a >= b
        if on:
            gate_cats.add(c)
        verdict = ("gate ON " if on else "gate off") + f" ({sig})"
        print(f"  {c:<14}{len(qs):>4}{b:>9.3f}{a:>9.3f}{a - b:>+9.3f}{nz:>9.3f}  {verdict}")
        summary["by_category"][c] = {
            "n": len(qs),
            "base": b,
            "agentic": a,
            "delta": a - b,
            "noise": nz,
        }

    def policy_mean(policy: str) -> tuple[float, int]:
        scores = []
        for q in qids:
            if policy == "baseline":
                use = False
            elif policy == "all-agentic":
                use = True
            elif policy == "oracle-category":
                use = baseline[q]["category"] in gate_cats
            else:  # oracle-perquery (unreachable upper bound)
                use = agentic[q]["score"] > baseline[q]["score"]
            scores.append(agentic[q]["score"] if use else baseline[q]["score"])
        return _macro(scores)

    print(f"\n  gate categories (agentic >= baseline): {sorted(gate_cats)}")
    base_mean = policy_mean("baseline")[0]
    overall_noise = _noise(len(qids))
    print(f"  overall judge noise band at n={len(qids)}: +-{overall_noise:.3f}\n")
    print(f"  {'policy':<18}{'mean':>9}{'vs base':>10}")
    for p in ["baseline", "all-agentic", "oracle-category", "oracle-perquery"]:
        m, _ = policy_mean(p)
        print(f"  {p:<18}{m:>9.3f}{m - base_mean:>+10.3f}")
        summary["policies"][p] = m
    summary["overall_noise"] = overall_noise

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
