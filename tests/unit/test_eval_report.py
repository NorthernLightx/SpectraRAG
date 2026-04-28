"""Eval reporter: JSON snapshot + readable markdown."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.eval.report import render_markdown, write_run_json, write_run_markdown
from src.types import (
    EvalRun,
    GenerationMetrics,
    PerQueryResult,
    RetrievalMetrics,
)


def _build_run() -> EvalRun:
    now = datetime.now(UTC)
    return EvalRun(
        run_id="r1",
        started_at=now,
        finished_at=now,
        golden_set_name="phase1",
        golden_set_version="v1",
        config={"retriever": "pipeline", "rerank": True},
        per_query=[
            PerQueryResult(
                query_id="q1",
                category="factual",
                text="What is X?",
                retrieved_chunk_ids=["c1", "c2"],
                retrieval=RetrievalMetrics(ndcg_at_5=0.9, recall_at_10=1.0, mrr=1.0),
                generation=GenerationMetrics(citation_rate=1.0),
                answer_text="X.",
                cited_chunk_ids=["c1"],
                latency_ms=200,
                tokens_in=80,
                tokens_out=20,
            ),
            PerQueryResult(
                query_id="q2_oc",
                category="out_of_corpus",
                text="?",
                retrieved_chunk_ids=["c5"],
                retrieval=RetrievalMetrics(ndcg_at_5=0.0, recall_at_10=0.0, mrr=0.0),
                latency_ms=180,
            ),
        ],
    )


def test_write_run_json_round_trips(tmp_path: Path) -> None:
    run = _build_run()
    path = tmp_path / "runs" / "run.json"
    write_run_json(run, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "r1"
    restored = EvalRun.model_validate(payload)
    assert restored.golden_set_name == "phase1"
    assert len(restored.per_query) == 2


def test_render_markdown_contains_expected_sections() -> None:
    md = render_markdown(_build_run())
    assert "# Eval Report — phase1 v1" in md
    assert "## Configuration" in md
    assert "## Retrieval (in-corpus queries)" in md
    assert "## Generation" in md
    assert "## Latency" in md
    assert "## Per-Query Results" in md
    # In-corpus only contributes to averages: q1 nDCG@5=0.9, so mean = 0.9000
    assert "0.9000" in md
    # Q2 (out_of_corpus) excluded from retrieval averages, present in per-query table
    assert "q2_oc" in md


def test_render_markdown_handles_no_generation_metrics() -> None:
    """If no query has generation metrics, the Generation section is omitted."""
    run = _build_run()
    for q in run.per_query:
        q.generation = None
    md = render_markdown(run)
    assert "## Generation" not in md


def test_write_run_markdown_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "run.md"
    write_run_markdown(_build_run(), path)
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("# Eval Report")
