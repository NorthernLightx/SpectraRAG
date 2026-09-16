"""Adapter: MMDocIR annotations -> our GoldenSet YAML, plus a layout-label sidecar.

MMDocIR's per-question schema is `(Q, A, type, page_id, layout_mapping)`. Each row
maps to a GoldenQuery (src/types/eval.py) so it slots into the existing eval
harness, exactly as `build_mmlongbench_golden` does for MMLongBench-Doc.

Two things need care:

* **Page numbers.** MMDocIR `page_id` is 0-based; `relevant_pages` and the page
  chunk-ids everywhere in this repo are 1-based. Every page id is +1'd here.
  A handful of labels point past the end of their document; those are dropped,
  and a question left with no valid page is dropped with it.
* **Two type vocabularies.** Some questions carry MMLongBench-style evidence
  lists (`"['Chart']"`, `"['Table', 'Pure-text (Plain-text)']"`), others carry
  coarse tags (`text-only`, `multimodal-t`, `multimodal-f`, `meta-data`). Both
  are mapped onto ADR 0008's classifier vocabulary. `multimodal-t`/`multimodal-f`
  are read as table/figure evidence; the raw type is preserved in each query's
  `note` so that reading stays falsifiable against `MMDocIR_layouts.parquet`.

The layout (bbox) labels have no home in GoldenQuery (the schema is page-level),
so they are written to a sidecar JSON keyed by query_id rather than widening a
model the whole eval gate depends on.

Usage:
    .venv/Scripts/python.exe -m scripts.build_mmdocir_golden \\
        --annotations data/mmdocir/annotations.jsonl \\
        --manifest data/mmdocir/manifest.json \\
        --output data/golden/mmdocir-v1.yaml \\
        --layout-output data/golden/mmdocir-v1-layouts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

import src  # noqa: F401  -- loads .env
from scripts.fetch_mmdocir import _paper_id
from src.types.eval import GoldenQuery, GoldenSet

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def _categorize(raw_type: str) -> str:
    """MMDocIR modality type -> ADR 0008 classifier category.

    Figure/chart evidence wins over table when a question cites both, matching
    `build_mmlongbench_golden`'s precedence so the two golden sets stay
    comparable per-category.
    """
    lowered = raw_type.lower()
    if "chart" in lowered or "figure" in lowered or lowered == "multimodal-f":
        return "figure"
    if "table" in lowered or lowered == "multimodal-t":
        return "table"
    return "factual"


def _query_id(index: int, paper_id: str) -> str:
    """`mmdocir_0042_<paper>`, index-prefixed so ids are stable under re-runs
    of the same annotation file and sort in corpus order."""
    return f"mmdocir_{index:04d}_{paper_id[:30]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/mmdocir/annotations.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/mmdocir/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/golden/mmdocir-v1.yaml"))
    parser.add_argument(
        "--layout-output", type=Path, default=Path("data/golden/mmdocir-v1-layouts.json")
    )
    args = parser.parse_args()

    if not args.annotations.exists():
        raise SystemExit(f"Annotations not found: {args.annotations}; run scripts.fetch_mmdocir")
    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}; run scripts.fetch_mmdocir")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    # The golden set covers exactly the docs whose pages were fetched, so the
    # eval never scores a query whose gold page is not in the index.
    pages_by_paper = {doc["paper_id"]: int(doc["n_pages"]) for doc in manifest["docs"]}

    with args.annotations.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]

    queries: list[GoldenQuery] = []
    layout_labels: dict[str, list[dict[str, Any]]] = {}
    categories: Counter[str] = Counter()
    dropped_pages = 0
    dropped_queries = 0
    index = 0

    for row in rows:
        paper_id = _paper_id(str(row["doc_name"]))
        n_pages = pages_by_paper.get(paper_id)
        if n_pages is None:
            continue
        for question in row["questions"]:
            raw_type = str(question.get("type", ""))
            in_range = [int(p) for p in question.get("page_id", []) if 0 <= int(p) < n_pages]
            dropped_pages += len(question.get("page_id", [])) - len(in_range)
            if not in_range:
                dropped_queries += 1
                continue
            category = _categorize(raw_type)
            query_id = _query_id(index, paper_id)
            index += 1
            answer = str(question.get("A", "")).strip()
            queries.append(
                GoldenQuery(
                    query_id=query_id,
                    text=str(question["Q"]).strip(),
                    paper_id=paper_id,
                    category=category,
                    relevant_pages=sorted({p + 1 for p in in_range}),
                    expected_facts=[answer] if answer else [],
                    note=f"MMDocIR | domain={row['domain']} | type={raw_type}",
                )
            )
            categories[category] += 1
            mapping = [
                {
                    "page": int(entry["page"]) + 1,
                    "bbox": [float(v) for v in entry["bbox"]],
                    "page_size": [float(v) for v in entry["page_size"]],
                }
                for entry in question.get("layout_mapping", [])
                if 0 <= int(entry["page"]) < n_pages
            ]
            if mapping:
                layout_labels[query_id] = mapping

    golden = GoldenSet(name="mmdocir", version="v1", queries=queries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(golden.model_dump(mode="json"), fh, sort_keys=False, allow_unicode=True)

    args.layout_output.parent.mkdir(parents=True, exist_ok=True)
    args.layout_output.write_text(json.dumps(layout_labels, indent=1), encoding="utf-8")

    n_multipage = sum(1 for q in queries if len(q.relevant_pages) > 1)
    print(f"Wrote {len(queries)} queries to {args.output}")
    print(f"  categories: {dict(categories.most_common())}")
    print(f"  multi-page gold: {n_multipage}")
    print(f"  layout labels: {len(layout_labels)} queries -> {args.layout_output}")
    print(f"  dropped: {dropped_queries} queries, {dropped_pages} out-of-range page labels")


if __name__ == "__main__":
    main()
