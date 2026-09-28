"""The README's Results tables, recomputed from the committed runs they cite."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.derive_arms import derive, read_run
from scripts.rescore_mmlb_pages import rescore
from scripts.score_talk_retrieval import score_run
from src.eval.golden_set import load_golden_set
from src.eval.metrics_generation import answer_outcome, outcome_rates

_REPO = Path(__file__).resolve().parents[2]
_README = (_REPO / "README.md").read_text(encoding="utf-8")


def _yaml(path: str) -> Any:
    return yaml.safe_load((_REPO / path).read_text(encoding="utf-8"))


def _readme_numbers(label: str) -> list[str]:
    """The numbers after the first cell of the README table row whose first
    cell starts with `label`."""
    for line in _README.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells[0].startswith(label):
            return re.findall(r"\d+(?:\.\d+)?", " ".join(cells[1:]))
    raise AssertionError(f"README has no table row starting {label!r}")


def _page_recall(arm: dict[str, Any], golden: dict[str, Any], keep: set[str], k: int) -> float:
    """Mean recall over the first `k` distinct pages of each ranking."""
    cut = [
        {**pq, "retrieved_chunk_ids": _pages(pq["retrieved_chunk_ids"])[:k]}
        for pq in arm["per_query"]
    ]
    # rescore reports recall@10, which on at most 10 pages is recall@k.
    scored = rescore({**arm, "per_query": cut}, golden)["per_query"]
    values: list[float] = [
        pq["retrieval"]["recall_at_10"] for pq in scored if pq["query_id"] in keep
    ]
    return sum(values) / len(values)


def _pages(chunk_ids: list[str]) -> list[str]:
    """The distinct pages of a ranking, in rank order."""
    return list(dict.fromkeys("::".join(c.split("::")[:2]) for c in chunk_ids))


def test_mmdocir_table_matches_its_run() -> None:
    run = read_run(_REPO / "data/eval/mmdocir-depth50-legs.json.gz")
    golden = _yaml("data/golden/mmdocir-v1.yaml")
    # Reported on the questions whose documents are not in the MMLongBench corpus (docs/results.md).
    mmlb = {
        q["paper_id"]
        for q in _yaml("data/golden/mmlongbench-v1.yaml")["queries"]
        if q.get("paper_id")
    }
    keep = {q["query_id"] for q in golden["queries"] if q["paper_id"] not in mmlb}
    # The served search: each leg cut to ten results, fused with weight 1 (ADR 0032).
    for pq in run["per_query"]:
        pq["leg_chunk_ids"] = {leg: ids[:10] for leg, ids in pq["leg_chunk_ids"].items()}
    arms = derive(run, golden, top_k=10, weights=[1.0])

    assert re.search(rf"{len(keep):,}\s+questions", _README)
    for label, arm in (
        ("text only", "text-only"),
        ("text and page images", "hybrid-w1"),
        ("page images only", "visual-only"),
    ):
        expected = [f"{_page_recall(arms[arm], golden, keep, k):.2f}" for k in (5, 10)]
        assert _readme_numbers(label) == expected, label


@pytest.mark.parametrize(
    ("label", "run_path", "golden_path", "pages_dir", "fused"),
    [
        (
            "talks with slides",
            "data/eval/mcif-talks-legs.json.gz",
            "data/golden/mcif-v1.yaml",
            "data/eval/recordings/mcif",
            True,
        ),
        (
            "meetings",
            "data/eval/ami-meetings-text-v2.json.gz",
            "data/golden/qmsum-ami-test-v2.yaml",
            "data/eval/recordings/ami",
            False,
        ),
    ],
)
def test_recordings_table_matches_its_runs(
    label: str, run_path: str, golden_path: str, pages_dir: str, fused: bool
) -> None:
    run = read_run(_REPO / run_path)
    if fused:
        # A talk is searched like a PDF: both legs fused with weight 1 (ADR 0034).
        run = derive(run, _yaml(golden_path), top_k=10, weights=[1.0])["hybrid-w1"]
    golden = {
        q.query_id: q for q in load_golden_set(_REPO / golden_path).queries if q.relevant_spans
    }
    rows, _ = score_run(run, golden, _REPO / pages_dir, None)

    def mean(key: str) -> str:
        return f"{sum(r[key] for r in rows) / len(rows):.2f}"

    expected = [str(len(rows))] + [
        mean(key) for key in ("hit_at_1", "random_hit_at_1", "hit_at_3", "random_hit_at_3")
    ]
    assert _readme_numbers(label) == expected


@pytest.mark.parametrize(
    ("label", "run_path"),
    [
        ("within its own document", "data/eval/answers-mmdocir-gen150-scoped.json.gz"),
        ("across all 218 documents", "data/eval/answers-mmdocir-gen150-unscoped.json.gz"),
    ],
)
def test_answers_table_matches_its_runs(label: str, run_path: str) -> None:
    run = read_run(_REPO / run_path)
    outcomes = [
        answer_outcome(pq["answer_text"], pq["generation"]["answer_correctness"])
        for pq in run["per_query"]
    ]
    rates = outcome_rates(outcomes)

    assert len(outcomes) == 150
    assert _readme_numbers(label) == [f"{rates[k]:.2f}" for k in ("correct", "refused", "wrong")]
