"""Adapter: MMLongBench-Doc parquet -> our GoldenSet YAML.

MMLongBench-Doc's QA schema is `(doc_id, doc_type, question, answer,
evidence_pages, evidence_sources, answer_format)`. We map each row to a
GoldenQuery (src/types/eval.py) so it slots into the existing eval harness.

Category mapping follows ADR 0008's classifier vocabulary so the routing
analysis cross-references cleanly with our golden v3 numbers:
  * answer_format == "None" or empty evidence_sources -> out_of_corpus
  * any "Chart" or "Figure" in evidence_sources       -> figure
  * "Table" in evidence_sources (no Chart/Figure)     -> table
  * only Pure-text / Generalized-text                 -> factual
  * default                                            -> factual

Page-level only -- MMLongBench provides evidence_pages but no chunk-level
labels. We populate `relevant_pages` and leave `relevant_chunk_ids` empty.

Usage:
    .venv/Scripts/python.exe -m scripts.build_mmlongbench_golden \\
        --parquet data/mmlongbench/qa.parquet \\
        --pdfs data/mmlongbench/documents \\
        --output data/golden/mmlongbench-v1.yaml \\
        [--limit-docs 5]      # cap to N docs (alphabetical) for smoke testing
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

import src  # noqa: F401  -- loads .env

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def _parse_str_list(raw: str | None) -> list[str]:
    """Parse a Python-repr-formatted list-of-strings safely.

    The parquet stores strings like `"['Chart', 'Pure-text (Plain-text)']"`.
    We replace single quotes with double quotes and json.loads -- avoids the
    eval / literal_eval surface entirely. Returns [] on parse failure or
    empty input. The MMLongBench values don't contain embedded apostrophes,
    so this is safe for this dataset.
    """
    if not raw or raw.strip() in ("", "[]"):
        return []
    try:
        parsed = json.loads(raw.replace("'", '"'))
    except json.JSONDecodeError:
        return []
    return [str(x) for x in parsed] if isinstance(parsed, list) else []


def _parse_int_list(raw: str | None) -> list[int]:
    """Parse a Python-repr-formatted list-of-ints. Same trick as above."""
    if not raw or raw.strip() in ("", "[]"):
        return []
    try:
        parsed = json.loads(raw.replace("'", '"'))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out: list[int] = []
    for v in parsed:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


def _categorise(evidence_sources: list[str], answer_format: str | None) -> str:
    """Map MMLongBench evidence labels to our QueryCategory Literal."""
    if (answer_format == "None") or (not evidence_sources):
        return "out_of_corpus"
    sources_lower = {s.lower() for s in evidence_sources}
    has_figure = any(("figure" in s) or ("chart" in s) for s in sources_lower)
    has_table = any("table" in s for s in sources_lower)
    if has_figure:
        return "figure"
    if has_table:
        return "table"
    return "factual"


def _safe_paper_id(doc_id: str) -> str:
    return doc_id[: -len(".pdf")] if doc_id.endswith(".pdf") else doc_id


def _query_id(idx: int, doc_id: str) -> str:
    stem = _safe_paper_id(doc_id)
    return f"mmlb_{idx:04d}_{stem[:30]}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, default=Path("data/mmlongbench/qa.parquet"))
    p.add_argument("--pdfs", type=Path, default=Path("data/mmlongbench/documents"))
    p.add_argument("--output", type=Path, default=Path("data/golden/mmlongbench-v1.yaml"))
    p.add_argument(
        "--limit-docs",
        type=int,
        default=0,
        help="Cap to first N docs (alphabetical). 0 = all.",
    )
    p.add_argument(
        "--exclude-ooc",
        action="store_true",
        help="Drop out_of_corpus queries (smaller smoke set).",
    )
    args = p.parse_args()

    rows = pq.read_table(args.parquet).to_pylist()  # type: ignore[no-untyped-call]
    print(f"Loaded {len(rows)} QAs from {args.parquet}")

    # Filter to docs whose PDFs are locally available
    available = {f.name for f in args.pdfs.glob("*.pdf")}
    rows = [r for r in rows if r["doc_id"] in available]
    print(f"  {len(rows)} have a local PDF in {args.pdfs}")

    # Cap to first N docs if requested
    if args.limit_docs > 0:
        keep_docs = sorted({r["doc_id"] for r in rows})[: args.limit_docs]
        rows = [r for r in rows if r["doc_id"] in keep_docs]
        print(f"  --limit-docs {args.limit_docs}: kept {len(keep_docs)} docs, {len(rows)} queries")

    queries: list[dict[str, Any]] = []
    cat_counts: Counter[str] = Counter()
    for idx, r in enumerate(rows):
        sources = _parse_str_list(r.get("evidence_sources"))
        pages = _parse_int_list(r.get("evidence_pages"))
        category = _categorise(sources, r.get("answer_format"))
        if args.exclude_ooc and category == "out_of_corpus":
            continue
        cat_counts[category] += 1

        queries.append(
            {
                "query_id": _query_id(idx, r["doc_id"]),
                "text": r["question"],
                "paper_id": _safe_paper_id(r["doc_id"]),
                "category": category,
                "relevant_chunk_ids": [],
                "relevant_pages": pages,
                "expected_facts": [r["answer"]] if r.get("answer") else [],
                "note": (
                    f"MMLongBench-Doc | doc_type={r.get('doc_type')} | "
                    f"evidence_sources={sources!r} | answer_format={r.get('answer_format')}"
                ),
            }
        )

    out = {
        "name": "mmlongbench-doc",
        "version": "v1",
        "queries": queries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True, width=120)

    print(f"\nWrote {args.output}")
    print(f"  total queries: {len(queries)}")
    print("  category breakdown:")
    for cat, n in cat_counts.most_common():
        print(f"    {n:4d}  {cat}")


if __name__ == "__main__":
    main()
