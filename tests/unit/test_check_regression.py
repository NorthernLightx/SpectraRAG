"""Regression gate: macro-mean computation, threshold logic, exit codes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_regression import _compute_deltas, _macro_mean


def _per_query(
    qid: str,
    category: str,
    ndcg5: float,
    *,
    faith: float | None = None,
    ar: float | None = None,
    cp: float | None = None,
) -> dict[str, object]:
    gen: dict[str, object] = {}
    if faith is not None:
        gen["faithfulness"] = faith
    if ar is not None:
        gen["answer_relevance"] = ar
    if cp is not None:
        gen["context_precision"] = cp
    return {
        "query_id": qid,
        "category": category,
        "retrieval": {"ndcg_at_5": ndcg5, "recall_at_10": 1.0, "mrr": ndcg5},
        "generation": gen or None,
    }


def test_macro_mean_excludes_ooc_for_retrieval_metrics() -> None:
    per_query = [
        _per_query("q1", "factual", 1.0),
        _per_query("q2", "factual", 0.5),
        _per_query("q3", "out_of_corpus", 0.0),
    ]
    # Retrieval: in-corpus only → 1.0 + 0.5 / 2 = 0.75
    assert _macro_mean(per_query, "ndcg_at_5") == 0.75


def test_macro_mean_includes_ooc_for_generation_metrics() -> None:
    """RAGAS-style: faithfulness of OOC refusal is meaningful (no hallucination)."""
    per_query = [
        _per_query("q1", "factual", 1.0, faith=0.9),
        _per_query("q2", "factual", 1.0, faith=0.7),
        _per_query("q3", "out_of_corpus", 0.0, faith=1.0),
    ]
    # Generation: all queries → (0.9 + 0.7 + 1.0) / 3 ≈ 0.8667
    assert _macro_mean(per_query, "faithfulness") == pytest.approx(0.8667, abs=0.001)


def test_macro_mean_returns_none_when_metric_missing() -> None:
    per_query = [_per_query("q1", "factual", 1.0)]
    assert _macro_mean(per_query, "faithfulness") is None


def test_macro_mean_picks_up_generation_field_in_corpus_only() -> None:
    """Confirms the in-corpus path still works when no OOC query is present."""
    per_query = [
        _per_query("q1", "factual", 1.0, faith=0.9),
        _per_query("q2", "factual", 1.0, faith=0.7),
    ]
    assert _macro_mean(per_query, "faithfulness") == 0.8


def test_compute_deltas_flags_regression() -> None:
    baseline = {"per_query": [_per_query("q1", "factual", 0.8)]}
    candidate = {"per_query": [_per_query("q1", "factual", 0.7)]}
    deltas = _compute_deltas(baseline, candidate, ("ndcg_at_5",), threshold=0.05)
    [d] = deltas
    assert d.regressed is True
    assert d.delta_rel is not None and d.delta_rel < -0.05


def test_compute_deltas_passes_within_threshold() -> None:
    baseline = {"per_query": [_per_query("q1", "factual", 1.0)]}
    candidate = {"per_query": [_per_query("q1", "factual", 0.96)]}  # -4%
    deltas = _compute_deltas(baseline, candidate, ("ndcg_at_5",), threshold=0.05)
    [d] = deltas
    assert d.regressed is False


def test_compute_deltas_handles_missing_metric_as_not_applicable() -> None:
    baseline = {"per_query": [_per_query("q1", "factual", 1.0)]}
    candidate = {"per_query": [_per_query("q1", "factual", 1.0)]}
    deltas = _compute_deltas(baseline, candidate, ("faithfulness",), threshold=0.05)
    [d] = deltas
    assert d.regressed is False
    assert d.baseline is None and d.candidate is None


def _write(path: Path, runs: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"per_query": runs}), encoding="utf-8")
    return path


def test_cli_exits_zero_when_no_regression(tmp_path: Path) -> None:
    base = _write(tmp_path / "base.json", [_per_query("q1", "factual", 0.8)])
    cand = _write(tmp_path / "cand.json", [_per_query("q1", "factual", 0.85)])
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.check_regression",
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PASS" in proc.stdout


def test_cli_exits_one_when_metric_regresses(tmp_path: Path) -> None:
    base = _write(tmp_path / "base.json", [_per_query("q1", "factual", 0.8)])
    cand = _write(tmp_path / "cand.json", [_per_query("q1", "factual", 0.5)])
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.check_regression",
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stdout
    assert "FAIL" in proc.stdout
