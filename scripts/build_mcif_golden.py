"""Adapter: MCIF QA pairs -> golden candidates for a human to add time spans.

MCIF gives each question a human-written reference answer and a modality tag,
but no timestamps. Every QA pair becomes a candidate with empty
`relevant_spans`; a human fills them in, and `promote_candidates` moves the
finished entries into the golden set. Unanswerable questions (MCIF `NA`) need no
span and promote as they are.

The modality tag maps onto the classifier vocabulary: `V` (video only) reads as
figure evidence, `NA` as out of corpus, `A` and `AV` as factual. The raw tag and
the question's origin stay in `note`.

Usage:
    uv run python -m scripts.build_mcif_golden \\
        --candidates data/golden/_candidates/mcif-v1.yaml \\
        --golden data/golden/mcif-v1.yaml
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from scripts.fetch_mcif import QUESTIONS_FILE, McifSample, download, load_qa_samples
from src.types.eval import GoldenQuery, GoldenSet, QueryCategory

# Every long-form question in the fixed-prompt split carries this preamble.
_PROMPT_PREFIX = "Answer the following question concisely given the English content: "

_HEADER = """\
# MCIF golden candidates. A HUMAN adds the evidence of each answerable question
# as time spans in seconds, for example
#   relevant_spans: [{start_s: 83.0, end_s: 101.5}]
# then runs
#   uv run python -m scripts.promote_candidates \\
#       --candidates data/golden/_candidates/mcif-v1.yaml --into data/golden/mcif-v1.yaml
"""


def _question(prompt: str) -> str:
    if not prompt.startswith(_PROMPT_PREFIX):
        raise ValueError(f"unexpected MCIF prompt: {prompt[:80]!r}")
    return prompt[len(_PROMPT_PREFIX) :].strip()


def _category(qa_type: str) -> QueryCategory:
    if qa_type == "V":
        return "figure"
    if qa_type == "NA":
        return "out_of_corpus"
    return "factual"


def to_candidates(samples: Sequence[McifSample], prompts: Mapping[str, str]) -> list[GoldenQuery]:
    """One candidate per QA sample, joined to its question by sample id."""
    candidates: list[GoldenQuery] = []
    for sample in samples:
        category = _category(sample.qa_type)
        candidates.append(
            GoldenQuery(
                query_id=f"mcif_{sample.iid}",
                text=_question(prompts[sample.id]),
                paper_id=sample.talk,
                category=category,
                expected_facts=[] if category == "out_of_corpus" else [sample.reference],
                note=f"MCIF | qa_type={sample.qa_type} | origin={sample.qa_origin}",
            )
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path, default=Path("data/golden/_candidates/mcif-v1.yaml")
    )
    parser.add_argument("--golden", type=Path, default=Path("data/golden/mcif-v1.yaml"))
    args = parser.parse_args()

    rows = pq.read_table(  # type: ignore[no-untyped-call]
        download(QUESTIONS_FILE), columns=["id", "prompt_en"]
    ).to_pylist()
    prompts = {str(r["id"]): str(r["prompt_en"]) for r in rows}
    candidates = to_candidates(load_qa_samples(), prompts)

    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(
        [c.model_dump(mode="json") for c in candidates],
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
    args.candidates.write_text(_HEADER + body, encoding="utf-8")

    # Promotion appends to an existing set; create it empty once, never overwrite.
    if not args.golden.exists():
        empty = GoldenSet(name="mcif", version="v1", queries=[])
        args.golden.write_text(
            yaml.safe_dump(empty.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )

    categories = Counter(c.category for c in candidates)
    print(f"Wrote {len(candidates)} candidates over {len({c.paper_id for c in candidates})} talks")
    print(f"  categories: {dict(categories.most_common())}")
    print(f"  -> {args.candidates}")


if __name__ == "__main__":
    main()
