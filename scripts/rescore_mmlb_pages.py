"""Re-score an MMLongBench run JSON at PAGE granularity, paper-aware.

The eval runner scores retrieval against `relevant_chunk_ids`, but
MMLongBench-Doc labels are page-level — every per-query `relevant_chunk_ids`
is empty in the golden YAML, so eval_run.py records 0.0 for nDCG/recall/MRR
on every query. This script reads the run JSON, maps each retrieved chunk-id
to its (paper, page) key (`paper::pN::cM` -> `(paper, N)`), scores against
the golden's `relevant_pages` paired with the query's `paper_id`, and writes
a copy with the per-query metrics filled in. Scoring is paper-aware: a
retrieved "page N" chunk only counts as a hit when it comes from the gold
paper, since retrieval is not paper-scoped (a query can surface chunks from
any corpus paper) and a same-numbered page in the wrong paper is not relevant.

That copy is what `data/eval/baseline-mmlongbench.json` is built from, and
what `scripts/check_regression.py --baseline` consumes for the multi-modal
regression gate. The original run JSON is left untouched — it remains the
faithful eval_run.py output (for audit) and this script the canonical
post-process.

Usage:

    .venv/Scripts/python.exe -m scripts.rescore_mmlb_pages \
        --run data/eval/runs/run-20260506-024716.json \
        --golden data/golden/mmlongbench-v1.yaml \
        --output data/eval/baseline-mmlongbench.json

Prints aggregate page-level macro means; matches `logs/rescore_mmlb.py` (the
ad-hoc predecessor) and the numbers cited in README.md "Why multi-modal?".
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


_PAGE_RE = re.compile(r"::p(\d+)")

# A page identity is (paper_id, page_number). The paper_id is the chunk-id
# prefix before the first "::"; relevance is keyed on the same tuple so a
# wrong-paper page never counts as a hit.
Page = tuple[str, int]


def _page_of(chunk_id: str) -> Page | None:
    match = _PAGE_RE.search(chunk_id)
    if match is None:
        return None
    return chunk_id.split("::", 1)[0], int(match.group(1))


def _dedup_pages_in_rank(retrieved_chunks: list[str]) -> list[Page]:
    """Walk retrieved chunks in rank order; keep each page on first appearance."""
    seen: set[Page] = set()
    pages: list[Page] = []
    for chunk_id in retrieved_chunks:
        page = _page_of(chunk_id)
        if page is None or page in seen:
            continue
        seen.add(page)
        pages.append(page)
    return pages


def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def _ndcg_at_k(retrieved_pages: list[Page], relevant: set[Page], k: int = 5) -> float:
    if not relevant:
        return 0.0
    rels = [1 if p in relevant else 0 for p in retrieved_pages[:k]]
    ideal = sorted(rels, reverse=True)
    ideal_dcg = _dcg(ideal)
    return _dcg(rels) / ideal_dcg if ideal_dcg > 0 else 0.0


def _recall_at_k(retrieved_pages: list[Page], relevant: set[Page], k: int = 10) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for p in retrieved_pages[:k] if p in relevant)
    return hits / len(relevant)


def _mrr(retrieved_pages: list[Page], relevant: set[Page]) -> float:
    if not relevant:
        return 0.0
    for idx, page in enumerate(retrieved_pages):
        if page in relevant:
            return 1.0 / (idx + 1)
    return 0.0


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def rescore(run: dict[str, Any], golden: dict[str, Any]) -> dict[str, Any]:
    """Returns a deep-ish copy of `run` with per_query[i].retrieval populated
    at page granularity. Keeps every other field intact so the file remains a
    drop-in for `check_regression.py`.

    Queries whose golden entry has no `relevant_pages` (label-gap or OOC)
    receive `retrieval` values of None — `check_regression._macro_mean` skips
    those queries, keeping the macro means honest. We explicitly do NOT
    mutate the `category` field, since the eval runner's category labels are
    a separate concern from page-level scoring fidelity.
    """
    relevant_by_qid: dict[str, set[Page]] = {
        q["query_id"]: {
            (q["paper_id"], page)
            for page in (q.get("relevant_pages") or [])
            if q.get("paper_id")
        }
        for q in golden["queries"]
    }
    out = dict(run)
    rescored_per_query: list[dict[str, Any]] = []
    for pq in run.get("per_query", []):
        record = dict(pq)
        relevant = relevant_by_qid.get(pq["query_id"], set())
        retrieved_pages = _dedup_pages_in_rank(pq.get("retrieved_chunk_ids") or [])
        if relevant:
            record["retrieval"] = {
                "ndcg_at_5": _ndcg_at_k(retrieved_pages, relevant, k=5),
                "recall_at_10": _recall_at_k(retrieved_pages, relevant, k=10),
                "mrr": _mrr(retrieved_pages, relevant),
            }
        else:
            record["retrieval"] = {
                "ndcg_at_5": None,
                "recall_at_10": None,
                "mrr": None,
            }
        rescored_per_query.append(record)
    out["per_query"] = rescored_per_query
    return out


def _print_summary(rescored: dict[str, Any]) -> None:
    """Prints the same macro means `check_regression.py` will compute on the
    output file — useful sanity check that the gate agrees with the rescore."""
    scored = [
        pq
        for pq in rescored["per_query"]
        if pq.get("category") != "out_of_corpus"
        and (pq.get("retrieval") or {}).get("recall_at_10") is not None
    ]
    ndcg = _avg([float(pq["retrieval"]["ndcg_at_5"]) for pq in scored])
    recall = _avg([float(pq["retrieval"]["recall_at_10"]) for pq in scored])
    mrr = _avg([float(pq["retrieval"]["mrr"]) for pq in scored])
    print(f"  Queries scored    : {len(scored)}")
    print(f"  nDCG@5    (macro) : {ndcg:.4f}")
    print(f"  recall@10 (macro) : {recall:.4f}")
    print(f"  MRR       (macro) : {mrr:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = json.loads(args.run.read_text(encoding="utf-8"))
    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    rescored = rescore(run, golden)

    print(f"Rescored run {run.get('run_id', args.run.name)} at page granularity.")
    _print_summary(rescored)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rescored, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
