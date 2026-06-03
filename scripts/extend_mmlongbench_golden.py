"""Extend the MMLongBench golden to the full benchmark, as a strict superset.

The eval set is small (149 queries, n~107 scored), which is why levers keep
landing "directional, not significant" (ADR 0019/0023/0025). MMLongBench ships
1082 questions over 134 locally-available docs, with human-authored gold answers
and evidence pages — so growing coverage adds statistical power WITHOUT the
machine authoring any ground truth (same provenance as mmlongbench-v1, just more
of it).

This builds mmlongbench-v2.yaml such that:
  - every v1 query is re-emitted byte-identical, with its query_id preserved, so
    the committed depth-50 dump / baselines / postret_failures (which key on
    `mmlb_NNNN`) stay valid against v2;
  - the remaining 933 questions are appended with fresh, non-colliding ids,
    built with the SAME mapping functions as build_mmlongbench_golden.py
    (imported, not reimplemented) so categorisation/labels can't drift.

Ingestion is separate and incremental: this only produces the labelled set.
A query is end-to-end scorable once its doc is in the retrieval collection.

Usage:
    .venv/Scripts/python.exe -m scripts.extend_mmlongbench_golden \
        --v1 data/golden/mmlongbench-v1.yaml \
        --output data/golden/mmlongbench-v2.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

import src  # noqa: F401  -- loads .env
from scripts.build_mmlongbench_golden import (
    _categorise,
    _parse_int_list,
    _parse_str_list,
    _query_id,
    _safe_paper_id,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def _entry(idx: int, r: dict[str, Any]) -> dict[str, Any]:
    """One golden entry from a parquet row — mirrors build_mmlongbench_golden."""
    sources = _parse_str_list(r.get("evidence_sources"))
    pages = _parse_int_list(r.get("evidence_pages"))
    return {
        "query_id": _query_id(idx, r["doc_id"]),
        "text": r["question"],
        "paper_id": _safe_paper_id(r["doc_id"]),
        "category": _categorise(sources, r.get("answer_format")),
        "relevant_chunk_ids": [],
        "relevant_pages": pages,
        "expected_facts": [r["answer"]] if r.get("answer") else [],
        "note": (
            f"MMLongBench-Doc | doc_type={r.get('doc_type')} | "
            f"evidence_sources={sources!r} | answer_format={r.get('answer_format')}"
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, default=Path("data/mmlongbench/qa.parquet"))
    p.add_argument("--pdfs", type=Path, default=Path("data/mmlongbench/documents"))
    p.add_argument("--v1", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    p.add_argument("--output", type=Path, default=Path("data/golden/mmlongbench-v2.yaml"))
    args = p.parse_args()

    v1 = yaml.safe_load(args.v1.read_text(encoding="utf-8"))
    v1_queries: list[dict[str, Any]] = v1["queries"]
    # Reuse a v1 query_id when (paper_id, question) matches — keys are unique in v1.
    v1_by_key = {(q["paper_id"], q["text"]): q for q in v1_queries}
    used_ids = {q["query_id"] for q in v1_queries}
    next_idx = max(int(q["query_id"].split("_")[1]) for q in v1_queries) + 1

    rows = pq.read_table(args.parquet).to_pylist()  # type: ignore[no-untyped-call]
    available = {f.name for f in args.pdfs.glob("*.pdf")}
    rows = [r for r in rows if r["doc_id"] in available]
    print(f"{len(rows)} available-PDF questions over {len({r['doc_id'] for r in rows})} docs")

    new_entries: list[dict[str, Any]] = []
    for r in rows:
        key = (_safe_paper_id(r["doc_id"]), r["question"])
        if key in v1_by_key:
            continue  # emitted verbatim from v1 below
        entry = _entry(next_idx, r)
        # _query_id embeds the doc stem, so two docs never collide; guard anyway.
        while entry["query_id"] in used_ids:
            next_idx += 1
            entry = _entry(next_idx, r)
        used_ids.add(entry["query_id"])
        new_entries.append(entry)
        next_idx += 1

    queries = v1_queries + new_entries  # v1 verbatim prefix, then the new tail

    # ---- Verification: superset integrity ----
    out_by_id = {q["query_id"]: q for q in queries}
    assert len(out_by_id) == len(queries), "duplicate query_id in v2"
    for q in v1_queries:
        assert out_by_id[q["query_id"]] == q, f"v1 entry {q['query_id']} not byte-identical in v2"
    matched = len(rows) - len(new_entries)
    assert matched == len(v1_queries), (
        f"v1 coverage mismatch: {matched} matched vs {len(v1_queries)}"
    )

    out = {"name": "mmlongbench-doc", "version": "v2", "queries": queries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True, width=120)

    cats = Counter(q["category"] for q in queries)
    in_corpus = sum(n for c, n in cats.items() if c != "out_of_corpus")
    print(f"\nWrote {args.output}")
    print(f"  v1 preserved : {len(v1_queries)} (query_ids + entries identical)")
    print(f"  appended     : {len(new_entries)}")
    print(f"  total        : {len(queries)}  ({in_corpus} in-corpus, {cats['out_of_corpus']} OOC)")
    print(
        f"  power vs v1  : in-corpus {len(v1_queries) - 40} -> {in_corpus}  (~{in_corpus / (len(v1_queries) - 40):.1f}x)"
    )
    print("  categories   :")
    for c, n in cats.most_common():
        print(f"    {n:4d}  {c}")


if __name__ == "__main__":
    main()
