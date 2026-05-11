"""Smoke-eval pre-flight: prereq checks + command shape (no real subprocess)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.smoke_eval import _build_eval_command, _check_prereqs


def test_prereqs_pass_when_paper_and_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """All required prereqs satisfied → empty error list (green light)."""
    monkeypatch.setenv("RAG_OPENROUTER_API_KEY", "sk-test")
    errors = _check_prereqs()
    # In CI the smoke paper might be absent; we don't assert empty here, only
    # that the API-key check passed (no message about it).
    assert not any("OPENROUTER_API_KEY" in e for e in errors)


def test_prereqs_flag_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    errors = _check_prereqs()
    assert any("OPENROUTER_API_KEY" in e for e in errors)


def test_command_includes_full_production_stack() -> None:
    """The built command must exercise all Tier 1 + Tier 0 features so smoke
    catches regressions in any of them — refusal gate, length-norm, region
    boost, paper-id filter, router, rerank, generate, judge."""
    cmd = _build_eval_command(pdf=Path("/tmp/p.pdf"))
    cmd_str = " ".join(cmd)
    assert "--rerank" in cmd
    assert "--rerank-length-norm" in cmd
    assert "--router" in cmd
    assert "--paper-id-filter" in cmd
    assert "--region-number-boost" in cmd
    assert "--refusal-score-threshold" in cmd
    assert "0.105" in cmd  # calibrated default ships in smoke
    assert "--generate" in cmd
    assert "--judge" in cmd
    assert "openrouter" in cmd_str  # both generator + judge use OpenRouter
    assert "--postgres-dsn" in cmd
    # Pass an empty string explicitly so smoke doesn't require Postgres.
    assert cmd[cmd.index("--postgres-dsn") + 1] == ""


def test_command_uses_dedicated_collection() -> None:
    """Smoke must not pollute the real eval collection."""
    cmd = _build_eval_command(pdf=Path("/tmp/p.pdf"))
    assert "smoke_eval_preflight" in cmd


def test_command_pdf_arg_is_caller_supplied() -> None:
    """The --pdf positional value mirrors what the caller passed."""
    cmd = _build_eval_command(pdf=Path("data/papers/custom.pdf"))
    pdf_idx = cmd.index("--pdf") + 1
    assert cmd[pdf_idx] == "data/papers/custom.pdf" or cmd[pdf_idx].endswith("custom.pdf")
