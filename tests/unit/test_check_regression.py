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


def test_config_note_names_the_changed_retrieval_knobs() -> None:
    from scripts.check_regression import config_note

    def run(fp: str, reranker: str) -> dict[str, object]:
        return {
            "config": {
                "retrieval_fingerprint": fp,
                "retrieval_config": {"reranker_model": reranker, "candidate_pool": 50},
            }
        }

    assert config_note(run("a", "x"), run("a", "x")) is None
    note = config_note(run("a", "x"), run("b", "y"))
    assert note is not None
    assert "reranker_model: 'x' -> 'y'" in note
    assert "candidate_pool" not in note
    assert "older run" in (config_note({"config": {}}, run("b", "y")) or "")


def test_per_query_losses_catch_a_drop_the_mean_absorbs() -> None:
    from scripts.check_regression import per_query_losses

    def run(values: list[float]) -> dict[str, object]:
        return {
            "per_query": [
                {"query_id": f"q{i}", "category": "factual", "retrieval": {"recall_at_10": v}}
                for i, v in enumerate(values)
            ]
            + [{"query_id": "ooc", "category": "out_of_corpus", "retrieval": {"recall_at_10": 1.0}}]
        }

    baseline = run([1.0] * 30 + [0.5])
    candidate = run([1.0] * 29 + [0.0, 1.0])
    # One query fell from found to missed; another rose. The mean moves 0.016.
    assert per_query_losses(baseline, candidate, "recall_at_10", 0.0) == [("q29", 1.0, 0.0)]
    assert per_query_losses(baseline, candidate, "recall_at_10", 1.0) == []


def test_per_query_flag_fails_the_gate(tmp_path: Path) -> None:
    def write(name: str, value: float) -> Path:
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "per_query": [
                        {
                            "query_id": "a",
                            "category": "factual",
                            "retrieval": {"recall_at_10": 1.0},
                        },
                        {
                            "query_id": "b",
                            "category": "factual",
                            "retrieval": {"recall_at_10": value},
                        },
                    ]
                    + [
                        {
                            "query_id": f"x{i}",
                            "category": "factual",
                            "retrieval": {"recall_at_10": 1.0},
                        }
                        for i in range(40)
                    ]
                }
            ),
            encoding="utf-8",
        )
        return path

    cmd = [
        sys.executable,
        "-m",
        "scripts.check_regression",
        "--baseline",
        str(write("b.json", 1.0)),
        "--candidate",
        str(write("c.json", 0.0)),
        "--metrics",
        "recall_at_10",
    ]
    assert subprocess.run(cmd, capture_output=True, check=False).returncode == 0
    result = subprocess.run(
        [*cmd, "--per-query", "recall_at_10"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 1
    assert "recall_at_10 dropped on b: 1.0000 -> 0.0000" in result.stdout


def test_a_query_missing_from_the_candidate_is_a_loss() -> None:
    from scripts.check_regression import per_query_losses

    def row(qid: str) -> dict[str, object]:
        return {"query_id": qid, "category": "factual", "retrieval": {"recall_at_10": 1.0}}

    losses = per_query_losses(
        {"per_query": [row("q1"), row("q2")]}, {"per_query": [row("q1")]}, "recall_at_10", 0.0
    )
    assert [q for q, _, _ in losses] == ["q2"]
