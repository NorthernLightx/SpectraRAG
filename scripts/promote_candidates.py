"""Promote human-filled golden candidates into a real golden set.

Pairs with `harvest_candidates.py`. Reads a `_candidates/*.yaml`, and for
each entry: validates it against the `GoldenQuery` model **and** asserts
the human actually filled the truth fields, then appends accepted entries
to the target `data/golden/<set>.yaml`. Stubs still left as TODO/blank are
rejected — the machine never ships an unlabeled golden.

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

    Raises ``ValidationError`` for bad types (e.g. ``category="TODO"`` is
    not a valid ``QueryCategory``) and ``NotLabeledError`` for an
    otherwise-valid but still-empty stub.
    """
    q = GoldenQuery.model_validate(d)
    if q.paper_id in ("", "TODO"):
        raise NotLabeledError(f"{q.query_id}: paper_id unset")
    if not q.expected_facts:
        raise NotLabeledError(f"{q.query_id}: expected_facts empty")
    if not (q.relevant_chunk_ids or q.relevant_pages):
        raise NotLabeledError(f"{q.query_id}: no relevant_chunk_ids / relevant_pages")
    return q


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
    doc.setdefault("queries", []).extend(accepted)
    args.into.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    print(f"promoted {len(accepted)} into {args.into}; rejected {len(rejected)} unlabeled/invalid.")


if __name__ == "__main__":
    main()
