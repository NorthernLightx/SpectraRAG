"""Promote human-filled golden candidates into a real golden set.

Pairs with `harvest_candidates.py`. Reads a `_candidates/*.yaml`, and for
each entry: validates it against the `GoldenQuery` model **and** asserts
the human actually filled the truth fields, then appends accepted entries
to the target `data/golden/<set>.yaml`. Stubs still left as TODO/blank are
rejected: the machine never ships an unlabeled golden.

    python -m scripts.promote_candidates \\
        --candidates data/golden/_candidates/candidates-<ts>.yaml \\
        --into data/golden/v3.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.types.eval import GoldenQuery


class NotLabeledError(ValueError):
    """Candidate is still a stub: a required truth field is unfilled."""


def _validate_candidate(d: dict[str, Any]) -> GoldenQuery:
    """Pydantic-validate + require a human to have filled the ground truth.

    Raises ``ValidationError`` for bad types (for example ``category="TODO"``
    is not a valid ``QueryCategory``) and ``NotLabeledError`` for an
    otherwise-valid but still-empty stub.
    """
    q = GoldenQuery.model_validate(d)
    if q.paper_id in ("", "TODO"):
        raise NotLabeledError(f"{q.query_id}: paper_id unset")
    # An unanswerable query's truth is the absence of facts and evidence.
    if q.category == "out_of_corpus":
        return q
    if not q.expected_facts:
        raise NotLabeledError(f"{q.query_id}: expected_facts empty")
    if not (q.relevant_chunk_ids or q.relevant_pages or q.relevant_spans):
        raise NotLabeledError(
            f"{q.query_id}: no relevant_chunk_ids / relevant_pages / relevant_spans"
        )
    return q


def _merge(
    existing: list[dict[str, Any]], accepted: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Append accepted queries whose id the set lacks. An id already in the set
    is skipped, not replaced: a correction made in the golden wins over a stale
    candidate."""
    have = {q.get("query_id") for q in existing}
    merged = list(existing)
    skipped: list[str] = []
    for q in accepted:
        if q["query_id"] in have:
            skipped.append(q["query_id"])
            continue
        have.add(q["query_id"])
        merged.append(q)
    return merged, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="Promote filled golden candidates.")
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--into", type=Path, required=True, help="target data/golden/<set>.yaml")
    args = ap.parse_args()

    raw = yaml.safe_load(args.candidates.read_text(encoding="utf-8")) or []
    accepted: list[dict[str, Any]] = []
    rejected: list[str] = []
    for d in raw:
        try:
            q = _validate_candidate(d)
        except (NotLabeledError, ValidationError) as exc:
            qid = d.get("query_id", "?") if isinstance(d, dict) else "?"
            rejected.append(f"{qid}: {type(exc).__name__}")
            continue
        accepted.append(q.model_dump())

    if not accepted:
        print(f"0 promoted; {len(rejected)} still-stub/invalid:")
        for r in rejected:
            print(f"  - {r}")
        sys.exit(1)

    doc = yaml.safe_load(args.into.read_text(encoding="utf-8"))
    doc["queries"], skipped = _merge(doc.get("queries") or [], accepted)
    args.into.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    print(
        f"promoted {len(accepted) - len(skipped)} into {args.into}; "
        f"{len(skipped)} already there; rejected {len(rejected)} unlabeled/invalid."
    )


if __name__ == "__main__":
    main()
