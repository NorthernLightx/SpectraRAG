"""Offline rebaseline: recompute retrieval metrics under updated goldens."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.rebaseline_offline import rebaseline


def _write_golden(path: Path, queries: list[dict[str, object]]) -> None:
    path.write_text(
        yaml.safe_dump({"name": "test", "version": "v0", "queries": queries}),
        encoding="utf-8",
    )


def _write_run(path: Path, per_query: list[dict[str, object]]) -> None:
    run = {
        "run_id": "test123",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        "golden_set_name": "test",
        "golden_set_version": "v0",
        "config": {},
        "per_query": per_query,
    }
    path.write_text(json.dumps(run), encoding="utf-8")


def test_rebaseline_promotes_retrieval_when_relevant_set_expands(tmp_path: Path) -> None:
    """Adding a chunk to relevant_chunk_ids that the run already retrieved
    must lift nDCG / recall / MRR — that's the whole point of this script."""
    golden = tmp_path / "v.yaml"
    _write_golden(
        golden,
        [
            {
                "query_id": "q1",
                "text": "x",
                "paper_id": "p1",
                "category": "table",
                # Both text-c5 and table-tab1 are now valid answers.
                "relevant_chunk_ids": ["p1::p2::c5", "p1::p2::tab1"],
            }
        ],
    )

    run_path = tmp_path / "run.json"
    # The run retrieved tab1 at rank 1, c5 at rank 2.
    _write_run(
        run_path,
        [
            {
                "query_id": "q1",
                "category": "table",
                "text": "x",
                "retrieved_chunk_ids": ["p1::p2::tab1", "p1::p2::c5", "p1::p3::c10"],
                "retrieval": {"ndcg_at_5": 0.5, "recall_at_10": 0.5, "mrr": 0.5},  # stale numbers
                "generation": None,
                "answer_text": None,
                "cited_chunk_ids": [],
                "latency_ms": 100,
                "tokens_in": 0,
                "tokens_out": 0,
            }
        ],
    )

    out = tmp_path / "rebaselined.json"
    new_run = rebaseline(run_path, golden, out)

    pq = new_run["per_query"][0]
    # Both relevant chunks retrieved at ranks 1 and 2 → recall=1.0, MRR=1.0,
    # nDCG@5 should be high (close to ideal since both labeled chunks are top-2).
    assert pq["retrieval"]["recall_at_10"] == pytest.approx(1.0)
    assert pq["retrieval"]["mrr"] == pytest.approx(1.0)
    assert pq["retrieval"]["ndcg_at_5"] > 0.9


def test_rebaseline_preserves_generation_metrics(tmp_path: Path) -> None:
    """faithfulness / answer_relevance / context_precision / citation_rate
    must pass through unchanged — they don't depend on relevant_chunk_ids."""
    golden = tmp_path / "v.yaml"
    _write_golden(
        golden,
        [{"query_id": "q1", "text": "x", "paper_id": "p1", "category": "factual",
          "relevant_chunk_ids": ["p1::p2::c5"]}],
    )

    run_path = tmp_path / "run.json"
    _write_run(
        run_path,
        [
            {
                "query_id": "q1",
                "category": "factual",
                "text": "x",
                "retrieved_chunk_ids": ["p1::p2::c5"],
                "retrieval": {"ndcg_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0},
                "generation": {
                    "citation_rate": 1.0,
                    "faithfulness": 0.95,
                    "answer_relevance": 0.88,
                    "context_precision": 0.77,
                },
                "answer_text": "...",
                "cited_chunk_ids": ["p1::p2::c5"],
                "latency_ms": 100,
                "tokens_in": 50,
                "tokens_out": 20,
            }
        ],
    )

    out = tmp_path / "rebaselined.json"
    new_run = rebaseline(run_path, golden, out)
    gen = new_run["per_query"][0]["generation"]
    assert gen["faithfulness"] == 0.95
    assert gen["answer_relevance"] == 0.88
    assert gen["context_precision"] == 0.77
    assert gen["citation_rate"] == 1.0


def test_rebaseline_keeps_ooc_at_zero(tmp_path: Path) -> None:
    """OOC queries are 0 by construction regardless of what's retrieved."""
    golden = tmp_path / "v.yaml"
    _write_golden(
        golden,
        [{"query_id": "q1", "text": "x", "paper_id": "p1",
          "category": "out_of_corpus", "relevant_chunk_ids": []}],
    )

    run_path = tmp_path / "run.json"
    _write_run(
        run_path,
        [
            {
                "query_id": "q1",
                "category": "out_of_corpus",
                "text": "x",
                "retrieved_chunk_ids": ["p1::p1::c0"],
                "retrieval": {"ndcg_at_5": 0.5, "recall_at_10": 0.5, "mrr": 0.5},
                "generation": None,
                "answer_text": None,
                "cited_chunk_ids": [],
                "latency_ms": 100,
                "tokens_in": 0,
                "tokens_out": 0,
            }
        ],
    )

    out = tmp_path / "rebaselined.json"
    new_run = rebaseline(run_path, golden, out)
    pq = new_run["per_query"][0]
    assert pq["retrieval"]["ndcg_at_5"] == 0.0
    assert pq["retrieval"]["recall_at_10"] == 0.0
    assert pq["retrieval"]["mrr"] == 0.0


def test_rebaseline_records_provenance(tmp_path: Path) -> None:
    """The output JSON must record what golden was applied + the original run id
    so the rebaselined file is self-describing."""
    golden = tmp_path / "v.yaml"
    _write_golden(
        golden,
        [{"query_id": "q1", "text": "x", "paper_id": "p1",
          "category": "factual", "relevant_chunk_ids": ["p1::p1::c0"]}],
    )

    run_path = tmp_path / "run.json"
    _write_run(
        run_path,
        [
            {
                "query_id": "q1",
                "category": "factual",
                "text": "x",
                "retrieved_chunk_ids": [],
                "retrieval": {"ndcg_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0},
                "generation": None,
                "answer_text": None,
                "cited_chunk_ids": [],
                "latency_ms": 0,
                "tokens_in": 0,
                "tokens_out": 0,
            }
        ],
    )

    out = tmp_path / "rebaselined.json"
    new_run = rebaseline(run_path, golden, out)
    assert new_run["rebaselined_from_run_id"] == "test123"
    assert "v.yaml" in new_run["rebaselined_against"]


def test_rebaseline_handles_query_absent_in_updated_goldens(tmp_path: Path) -> None:
    """If a run had a query that's been dropped from goldens, keep retrieval
    metrics as-is and add a note flag — don't crash."""
    golden = tmp_path / "v.yaml"
    _write_golden(
        golden,
        [{"query_id": "q_kept", "text": "y", "paper_id": "p1",
          "category": "factual", "relevant_chunk_ids": ["p1::p1::c0"]}],
    )

    run_path = tmp_path / "run.json"
    _write_run(
        run_path,
        [
            {
                "query_id": "q_dropped",
                "category": "factual",
                "text": "x",
                "retrieved_chunk_ids": ["p1::p1::c0"],
                "retrieval": {"ndcg_at_5": 0.7, "recall_at_10": 1.0, "mrr": 0.5},
                "generation": None,
                "answer_text": None,
                "cited_chunk_ids": [],
                "latency_ms": 0,
                "tokens_in": 0,
                "tokens_out": 0,
            }
        ],
    )
    out = tmp_path / "rebaselined.json"
    new_run = rebaseline(run_path, golden, out)
    pq = new_run["per_query"][0]
    # Untouched retrieval block.
    assert pq["retrieval"]["ndcg_at_5"] == 0.7
    assert "absent in updated goldens" in pq["rebaseline_note"]
