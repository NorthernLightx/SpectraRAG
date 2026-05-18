"""Harvest golden-candidate stubs from eval-run logs (see CONTRIBUTING "Scripts layout").

Reference-free triage. Reads `data/eval/runs/*.json` (no infra — works off
artifacts the eval already produces), flags interactions a human should
review (low judged metrics, false/missing refusal, empty retrieval), and
writes GoldenQuery-shaped *stubs* to `data/golden/_candidates/` with the
truth fields (paper_id / category / relevant_* / expected_facts) left
blank for a human.

It never invents ground truth — it proposes the question plus the model's
answer/retrieval as a review aid. `promote_candidates.py` is the
human-gated step that validates a filled stub and appends it to a real
golden set.

Heuristics are noisy by design (a flag means "look here", not a verdict)
and biased toward cases the *current* system is unsure about; pair the
harvest with a random sample before treating it as representative.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "eval" / "runs"
OUT_DIR = ROOT / "data" / "golden" / "_candidates"
_REFUSAL = (
    "not stated in the provided context",
    "cannot answer this question",
    "i cannot answer",
)
# Judged metrics are 0..1; below these = worth a human's eyes.
_FAITH, _AREL, _CPREC = 0.5, 0.5, 0.4


def _is_refusal(answer: str | None) -> bool:
    """Heuristic: does the answer look like a refusal phrase (lowercased)."""
    low = (answer or "").lower()
    return any(p in low for p in _REFUSAL)


def _flag_reasons(pq: dict[str, Any]) -> list[str]:
    """Reference-free reasons this logged query merits human review. Pure."""
    reasons: list[str] = []
    gen = pq.get("generation") or {}
    for key, thresh, label in (
        ("faithfulness", _FAITH, "low_faithfulness"),
        ("answer_relevance", _AREL, "low_answer_relevance"),
        ("context_precision", _CPREC, "low_context_precision"),
    ):
        v = gen.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v < thresh:
            reasons.append(f"{label}={float(v):.2f}")
    cat = pq.get("category")
    refused = _is_refusal(pq.get("answer_text"))
    if refused and cat != "out_of_corpus":
        reasons.append("false_refusal")  # said "can't" on an answerable query
    if not refused and cat == "out_of_corpus" and pq.get("answer_text"):
        reasons.append("missing_refusal")  # should have refused (hallucination risk)
    if not pq.get("retrieved_chunk_ids"):
        reasons.append("empty_retrieval")
    return reasons


def _to_candidate(pq: dict[str, Any], run_id: str) -> dict[str, Any]:
    """GoldenQuery-shaped stub; truth fields are placeholders for a human.

    `category`/`paper_id` use "TODO" deliberately — invalid, so
    `promote_candidates` rejects the stub until a human fills them.
    """
    answer = str(pq.get("answer_text") or "")
    return {
        "query_id": f"cand_{pq.get('query_id', 'unknown')}",
        "text": pq.get("text", ""),
        "paper_id": "TODO",
        "category": "TODO",
        "relevant_chunk_ids": [],
        "relevant_pages": [],
        "expected_facts": [],
        "note": (
            f"harvested run={run_id} flags={_flag_reasons(pq)} "
            f"model_answer={answer[:160]!r} "
            f"retrieved={list(pq.get('retrieved_chunk_ids') or [])[:6]}"
        ),
    }


def main() -> None:
    run_files = sorted(RUNS.glob("run-*.json")) + sorted(RUNS.glob("*/run-*.json"))
    candidates: list[dict[str, Any]] = []
    for rp in run_files:
        try:
            run = json.loads(rp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_id = str(run.get("run_id", rp.stem))
        for pq in run.get("per_query", []):
            if isinstance(pq, dict) and _flag_reasons(pq):
                candidates.append(_to_candidate(pq, run_id))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"candidates-{time.strftime('%Y%m%d-%H%M%S')}.yaml"
    out.write_text(
        "# Golden-candidate stubs. A HUMAN fills paper_id/category/"
        "relevant_*/expected_facts,\n# then runs `python -m "
        "scripts.promote_candidates`. The machine never authors ground truth.\n"
        + yaml.safe_dump(candidates, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    print(f"flagged {len(candidates)} candidate(s) from {len(run_files)} run file(s) -> {out}")


if __name__ == "__main__":
    main()
